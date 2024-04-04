import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphSAGE



class MyResnet101(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.resnet101(weights='IMAGENET1K_V2')  # Sử dụng pretrained ResNet-152
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
    def __init__(self, device):
        super(MyGCNModel, self).__init__()

        self.device = device
        self.in_features = 2048
        self.out_features = 2048
        self.num_layers = 2
        self.aggregation = "mean"

        self.convs = nn.ModuleList()
          # Define num_layers here
        num_layers = 2  # Example value

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GraphSAGE(self.in_features, self.out_features, num_layers=num_layers))  # Add num_layers argument
            else:
                self.convs.append(GraphSAGE(self.out_features, self.out_features, num_layers=num_layers))  # Add num_layers argument


    def forward(self, x, edge_index):
        x = x.to(self.device)  # Ensure data is on the correct device
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.aggregation == "mean":
                x = x.mean(dim=1)
        return x
    
    def extract_features(self, x, edge_index):
        x = x.to(self.device)  # Ensure data is on the correct device
        x = self.forward(x, edge_index)
        return x.cpu().detach().numpy()  # Return NumPy array for compatibility
