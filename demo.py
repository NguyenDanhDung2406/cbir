import time
import torch
import faiss
import pathlib
from PIL import Image

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

def retrieve_image(img, feature_extractor):
    if feature_extractor == 'VGG19':
        extractor = MyVGG19(device)
    elif feature_extractor == 'Resnet152':
        extractor = MyResnet152(device)
    elif feature_extractor == 'EfficientNetB7':
        extractor = MyEfficientNetB7(device)
    elif feature_extractor == 'AlexNet':
        extractor = MyAlexNet(device)
    elif feature_extractor == 'GCN':
        extractor = PretrainedGCNKNN(device)

    transform = get_transformation()

    img = img.convert('RGB')
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=10)

    return indices[0]

def main():
    st.title('CONTENT-BASED IMAGE RETRIEVAL')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ('Resnet152', 'VGG19', 'EfficientNetB7', 'AlexNet', 'GCN'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150, 150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option)
            image_list = get_image_list(image_root)

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            # Display results in a grid with 2 images per row
            for i in range(0, len(retriev), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(retriev):
                        image = Image.open(image_list[retriev[i + j]])
                        with cols[j]:
                            st.image(image, use_column_width='always')

if __name__ == '__main__':
    main()
