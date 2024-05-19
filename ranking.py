import os
import time
import pathlib
from PIL import Image
from argparse import ArgumentParser

import torch
import faiss

from src.feature_extraction import MyResnet152, MyVGG16, MyResnet101, MyGCNModel
from src.dataloader import get_transformation
import module_name 
import torchvision.transforms as transforms

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']

query_root = './dataset/groundtruth'
image_root = './dataset/paris'
feature_root = './dataset/feature'
evaluate_root = './dataset/evaluation'


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in os.listdir(image_root):
        image_list.append(image_path[:-4])
    image_list = sorted(image_list, key = lambda x: x)
    return image_list

def get_transformation():
    # Function to return the transformation for images
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet152')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--top_k", required=False, type=int, default=11)
    parser.add_argument("--crop", required=False, type=bool, default=False)

    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (args.feature_extractor == 'Resnet152'):
        extractor = MyResnet152(device)
    elif (args.feature_extractor == 'VGG16'):
        extractor = MyVGG16(device)
    elif (args.feature_extractor == 'Resnet101'):
        extractor = MyResnet101(device)
    elif (args.feature_extractor == 'GCN'):
        extractor = MyGCNModel(device)
    else:
        print("No matching model found")
        return

    img_list = get_image_list(image_root)
    print(f"Total images: {len(img_list)}")  # Debug: Print the total number of images
    transform = get_transformation()

    for path_file in os.listdir(query_root):
        if (path_file[-9:-4] == 'query'):
            rank_list = []

            with open(query_root + '/' + path_file, "r") as file:
                img_query, left, top, right, bottom = file.read().split()

            test_image_path = pathlib.Path('./dataset/paris/' + img_query + '.jpg')
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')

            path_crop = 'original'
            if (args.crop):
                pil_image=pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            feat = extractor.extract_features(image_tensor)

            indexer = faiss.read_index(feature_root + '/' + args.feature_extractor + '.index.bin')

            _, indices = indexer.search(feat, k=args.top_k)
            print(f"Indices: {indices[0]}")  # Debug: Print the indices returned by the search

            for index in indices[0]:
                if index < len(img_list):
                    rank_list.append(str(img_list[index]))
                else:
                    print(f"Warning: index {index} out of range for img_list of size {len(img_list)}")  # Debug: Warning message

            output_dir = os.path.join(evaluate_root, path_crop, args.feature_extractor)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, path_file[:-10] + '.txt')

            with open(output_file, "w") as file:
                file.write("\n".join(rank_list))

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()