import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch_geometric.nn import GCNConv


class MyResnet101(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.resnet101(pretrained=True)  # Sử dụng pretrained ResNet-152
        # Lấy các lớp của mô hình
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 2048  # Độ dài của vector đặc trưng

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
        image = transform(image)

        # Truyền ảnh qua mô hình Resnet-152 và lấy ra feature map của ảnh đầu vào
        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        # Trả về các đặc trưng dưới dạng mảng numpy
        return feature.cpu().detach().numpy()
    

class MyResnet152(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.resnet152(weights='IMAGENET1K_V2')  # Sử dụng pretrained ResNet-152
        # Lấy các lớp của mô hình
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 2048  # Độ dài của vector đặc trưng

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
        image = transform(image)

        # Truyền ảnh qua mô hình Resnet-152 và lấy ra feature map của ảnh đầu vào
        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        # Trả về các đặc trưng dưới dạng mảng numpy
        return feature.cpu().detach().numpy()
        
class MyVGG16(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 25088 # the length of the feature vector

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], 
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
        image = transform(image)

        # Feed the image into the model for feature extraction
        with torch.no_grad():
            feature = self.model.features(image)
            feature = torch.flatten(feature, start_dim=1)

        # Return features to numpy array
        return feature.cpu().detach().numpy()
    
class MyGCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, device):
        super(MyGCNModel, self).__init__()

        # Tạo một lớp GCN với 2048 đầu vào và 512 đầu ra
        self.conv1 = GCNConv(num_features, 512)
        # Tạo một lớp GCN với 512 đầu vào và số lớp đầu ra tương ứng với số lớp của dữ liệu của bạn
        self.conv2 = GCNConv(512, num_classes)
        
        self.device = device
        self.shape = num_classes  # Độ dài của vector đặc trưng

    def extract_features(self, x, edge_index):
        # Truyền dữ liệu qua lớp GCN đầu tiên
        x = F.relu(self.conv1(x, edge_index))
        # Truyền dữ liệu qua lớp GCN thứ hai và không áp dụng hàm kích hoạt ở đây
        x = self.conv2(x, edge_index)

        return x

    def forward(self, x, edge_index):
        # Khi bạn muốn sử dụng mô hình để dự đoán
        with torch.no_grad():
            x = self.extract_features(x, edge_index)
        return x.cpu().detach().numpy()