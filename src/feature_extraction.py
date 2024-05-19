import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image  # Import Image module from PIL package
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data



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
        self.model = self.model.features
        self.shape = 25088
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self.model.eval().to(device)

    def preprocess_image(self, image):
        # Kiểm tra xem image có phải là tensor không
        if not isinstance(image, torch.Tensor):
            # Chuyển đổi thành tensor nếu không phải là tensor
            image = self.transform(image).unsqueeze(0).to(self.device)
        else:
            image = image.to(self.device)

        return image

    def extract_features(self, image):
        image = self.preprocess_image(image)

        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        return feature.cpu().detach().numpy()
    
class MyGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class MyGCNModel(nn.Module):
    def __init__(self, device):
        super(MyGCNModel, self).__init__()
        self.device = device

        # Define a transformation layer to convert patches into features
        self.patch_to_feature = nn.Linear(3 * 16 * 16, 2048).to(device)

        # Define GCN
        self.gcn = MyGCN(input_dim=2048, hidden_dim=1024, output_dim=512).to(device)
        self.shape = 512  # Length of the feature vector after GCN

    def extract_features(self, image):
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).to(self.device)

        # Divide the image into patches (16x16)
        patches = image.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.contiguous().view(image.size(0), 3, -1, 16, 16)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, 16, 16)

        # Convert patches into features
        patches = patches.view(patches.size(0), -1).to(self.device)
        feature = self.patch_to_feature(patches)

        # Create edge_index for the image with the size of the patches
        edge_index = self.create_edge_index(image.size(2), image.size(3), 16).to(self.device)
        data = Data(x=feature, edge_index=edge_index)

        # Pass data through GCN
        gcn_feature = self.gcn(data.x, data.edge_index)

        # Pooling to create a single feature for the entire image
        gcn_feature = gcn_feature.view(image.size(0), -1, self.shape)  
        gcn_feature = gcn_feature.mean(dim=1)  

        # Return features as a numpy array
        return gcn_feature.cpu().detach().numpy()

    def create_edge_index(self, h, w, patch_size):
        edge_index = []
        num_patches_x = w // patch_size
        num_patches_y = h // patch_size

        def get_index(x, y):
            return y * num_patches_x + x

        for y in range(num_patches_y):
            for x in range(num_patches_x):
                current_index = get_index(x, y)
                if x > 0:
                    edge_index.append([current_index, get_index(x - 1, y)])  # left
                    edge_index.append([get_index(x - 1, y), current_index])  # bidirectional
                if x < num_patches_x - 1:
                    edge_index.append([current_index, get_index(x + 1, y)])  # right
                    edge_index.append([get_index(x + 1, y), current_index])  # bidirectional
                if y > 0:
                    edge_index.append([current_index, get_index(x, y - 1)])  # up
                    edge_index.append([get_index(x, y - 1), current_index])  # bidirectional
                if y < num_patches_y - 1:
                    edge_index.append([current_index, get_index(x, y + 1)])  # down
                    edge_index.append([get_index(x, y + 1), current_index])  # bidirectional

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Kiểm tra kích thước của edge_index
        if edge_index.numel() == 0:
            raise ValueError("edge_index is empty. Check the create_edge_index function.")
        print("edge_index size:", edge_index.size())
        
        return edge_index