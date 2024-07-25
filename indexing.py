import time
from argparse import ArgumentParser

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.feature_extraction import MyGCNModel, MyResnet152, MyVGG19, MyEfficientNetB7, MyAlexNet, PretrainedGCNKNN
from src.indexing import get_faiss_indexer
from src.dataloader import MyDataLoader

image_root = './dataset/paris'
feature_root = './dataset/feature'

def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet152')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--batch_size", required=False, type=int, default=64)

    print('Start indexing .......')
    start = time.time()

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    # Load module feature extraction 
    if args.feature_extractor == 'Resnet152':
        extractor = MyResnet152(device)
    elif args.feature_extractor == 'VGG19':
        extractor = MyVGG19(device)
    elif args.feature_extractor == 'EfficientNetB7':
        extractor = MyEfficientNetB7(device)
    elif args.feature_extractor == 'AlexNet':
        extractor = MyAlexNet(device)
    elif args.feature_extractor == 'GCN':
        extractor = PretrainedGCNKNN(device)
    else:
        print("No matching model found")
        return

    dataset = MyDataLoader(image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    indexer = get_faiss_indexer(extractor.shape)

    for images, image_paths in dataloader:
        images = images.to(device)
        features = extractor.extract_features(images)
        indexer.add(features)
    
    # Save features
    faiss.write_index(indexer, feature_root + '/' + args.feature_extractor + '.index.bin')
    
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()
