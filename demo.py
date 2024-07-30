import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd

import streamlit as st
from streamlit_cropper import st_cropper

from src.feature_extraction import MyGCNModel, MyVGG19, MyResnet152, MyEfficientNetB7, MyAlexNet, PretrainedGCNKNN
from src.dataloader import get_transformation

st.set_page_config(layout="wide")

device = torch.device('cuda:0')
image_root = './dataset/paris'
feature_root = './dataset/feature'


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: x.name)
    return image_list

def retrieve_image(feat, feature_extractor, query_vec=None, top_k=10):
    indexer = faiss.read_index(feature_root + '/' + feature_extractor + '.index.bin')

    if query_vec is None:
        _, indices = indexer.search(feat, k=top_k)
    else:
        _, indices = indexer.search(query_vec.reshape(1, -1), k=top_k)  # Tìm kiếm với query vector mới

    return indices[0]

def update_query_vector(query_vec, relevant_features, irrelevant_features, alpha=1, beta=0.75, gamma=0.15):
    """Cập nhật vector truy vấn sử dụng Rocchio Algorithm."""
    relevant_centroid = np.mean(relevant_features, axis=0)
    irrelevant_centroid = np.mean(irrelevant_features, axis=0)
    updated_query_vec = alpha * query_vec + beta * relevant_centroid - gamma * irrelevant_centroid
    return updated_query_vec

def process_image(img, extractor, transform):
    """Xử lý ảnh, trích xuất đặc trưng và trả về đặc trưng."""
    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    return feat

def main():
    st.title('CONTENT-BASED IMAGE RETRIEVAL')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ('Resnet152', 'VGG19', 'EfficientNetB7', 'AlexNet', 'GCN'))

        if option == 'VGG19':
            extractor = MyVGG19(device)
        elif option == 'Resnet152':
            extractor = MyResnet152(device)
        elif option == 'EfficientNetB7':
            extractor = MyEfficientNetB7(device)
        elif option == 'AlexNet':
            extractor = MyAlexNet(device)
        elif option == 'GCN':
            extractor = PretrainedGCNKNN(device)

        print(f"Extractor shape: {extractor.shape}")

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])
        
        # Initialize variables outside the if condition
        cropped_img = None
        feat = None
        
        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')

            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150, 150))
            st.image(cropped_img)

            # Xử lý ảnh và trích xuất đặc trưng
            transform = get_transformation()
            feat = process_image(cropped_img, extractor, transform)

    with col2:
        st.header('RESULT')
        if img_file:
            with st.spinner('Retrieving...'):
                start_time = time.time()
                indexer = faiss.read_index(feature_root + '/' + option + '.index.bin')
                print(f"Total vectors in index: {indexer.ntotal}")
                retrieval_results = retrieve_image(feat, option)  # Xóa tham số extractor=extractor
                image_list = get_image_list(image_root)
                end_time = time.time()

            st.success(f"Finished in {end_time - start_time:.2f} seconds")

            # Step 1: Hiển thị kết quả ban đầu với checkbox
            st.write("Initial Results:")
            relevant_images = []  # Danh sách các ảnh liên quan được chọn
            for i, result in enumerate(retrieval_results):
                image_path = pathlib.Path(image_root) / image_list[result].name
                try:
                    image = Image.open(image_path)
                except (IOError, OSError) as e:
                    st.error(f"Error loading image {image_path}: {e}")
                    continue  # Bỏ qua ảnh nếu không thể mở được

                col1, col2 = st.columns([3, 1])  # Chia cột để hiển thị ảnh và checkbox
                with col1:
                    st.image(image, use_column_width='always')
                with col2:
                    if st.checkbox("Relevant", key=f"checkbox_{i}"):
                        relevant_images.append(image_path)

            # Step 2 & 3: Relevance Feedback và cập nhật kết quả
            if st.button("Feedback") and relevant_images:
                # Load tất cả đặc trưng từ index
                try:
                    if isinstance(indexer, faiss.IndexFlatL2):
                        all_features = indexer.xb.copy()
                    else:
                        all_features = np.array([indexer.reconstruct(i) for i in range(indexer.ntotal)])
                    print(f"All features shape: {all_features}")
                except Exception as e:
                    st.error(f"Error retrieving features: {e}")
                    return
                relevant_features = []
                for x in relevant_images:
                    try:
                        idx = image_list.index(x)
                        if idx < len(all_features): # Kiểm tra xem chỉ số của ảnh có nằm trong khoảng của all_features không
                            relevant_features.append(all_features[idx])
                        else:
                            st.warning(f"Feature for image {x.name} not found in the index. Skipping...")
                    except ValueError:
                        st.warning(f"Image {x.name} not found in the index. Skipping...")
                        continue
                relevant_features = np.array(relevant_features)

                irrelevant_features = all_features[[image_list.index(x) for x in image_list if x not in relevant_images]]
                updated_query_vec = update_query_vector(feat.flatten(), relevant_features, irrelevant_features)

                 # Step 3: Tạo lại danh sách kết quả
                with st.spinner('Retrieving with feedback...'):
                    retrieval_results = retrieve_image(feat, option, query_vec=updated_query_vec)  # Truy vấn lại với updated_query_vec

                st.write("Results after feedback:")
                for i in range(0, len(retrieval_results), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(retrieval_results):
                            image = Image.open(image_list[retrieval_results[i + j]])
                            with cols[j]:
                                st.image(image, use_column_width='always')
            else:
                st.warning("No images selected for feedback.")

if __name__ == '__main__':
    image_list = get_image_list(image_root)
    main()