import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GATConv, global_max_pool,  knn_graph
from torchvision import models, transforms
import numpy as np
import faiss


class MyEfficientNetB7(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        # Sử dụng pretrained EfficientNet-B7
        self.model = models.efficientnet_b7(weights='IMAGENET1K_V1')
        
        # Lấy các lớp của mô hình, bỏ lớp phân loại cuối cùng
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 2560  # Độ dài của vector đặc trưng cho EfficientNet-B7

    def extract_features(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),  # Resize ảnh đầu vào nếu cần
            transforms.CenterCrop(224),  # Cắt ảnh trung tâm để kích thước phù hợp với EfficientNet-B7
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        # Thêm chiều batch cho ảnh
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Truyền ảnh qua mô hình EfficientNet và lấy ra feature map của ảnh đầu vào
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
        
class MyVGG19(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
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
    

class MyAlexNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        # Khởi tạo mô hình AlexNet pretrained
        self.model = models.alexnet(pretrained=True)

        # Loại bỏ lớp fully connected cuối cùng của AlexNet
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1]
        )

        # Số lượng đặc trưng cuối cùng
        self.shape = 4096

        # Thiết lập chế độ evaluation và di chuyển mô hình vào thiết bị
        self.model = self.model.eval().to(device)
        self.device = device

        # Định nghĩa các bước biến đổi ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize ảnh về kích thước 224x224
            transforms.ToTensor(),  # Chuyển ảnh về tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh theo ImageNet
        ])

    def preprocess_image(self, image):
        # Kiểm tra nếu ảnh không phải tensor thì chuyển về tensor và đưa lên thiết bị
        if not isinstance(image, torch.Tensor):
            image = self.transform(image).unsqueeze(0).to(self.device)
        else:
            image = image.to(self.device)
        return image

    def extract_features(self, image):
        image = self.preprocess_image(image)
        
        with torch.no_grad():
            features = self.model(image)
            features = torch.flatten(features, start_dim=1)

        return features.cpu().detach().numpy()


class MyGCNModel(nn.Module):
    def __init__(self, device):
        super(MyGCNModel, self).__init__()
        self.device = device
        
        self.conv1 = GCNConv(5, 64).to(device)
        self.conv2 = SAGEConv(64, 128).to(device)
        self.conv3 = GATConv(128, 256, heads=4, concat=False).to(device)
        self.conv4 = GCNConv(256, 512).to(device)
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm1d(64).to(device)
        self.bn2 = nn.BatchNorm1d(128).to(device)
        self.bn3 = nn.BatchNorm1d(256).to(device)
        self.bn4 = nn.BatchNorm1d(512).to(device)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Increase the size of the final feature vector
        self.fc = nn.Linear(1024, 1024).to(device)
        self.shape = 1024

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        
        # Sử dụng kết hợp nhiều loại pooling
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x1, x2], dim=1)
        
        x = self.fc(x)
        return x



    def extract_features(self, images):
        features = []
        for image in images:
            # Convert image to graph
            x, edge_index = self.image_to_graph(image)
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
            
            # Extract features
            with torch.no_grad():
                feature = self(x, edge_index, batch)
            features.append(feature.cpu().numpy().flatten())  # Flatten feature vector
        return np.array(features)

    def image_to_graph(self, image):
        height, width = image.shape[1:]
        x = image.view(3, -1).t()
        
        edge_index = []
        for i in range(height):
            for j in range(width):
                node = i * width + j
                # Kết nối 8 điểm lân cận
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            edge_index.append([node, ni * width + nj])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        # Thêm đặc trưng vị trí cho mỗi nút
        pos_enc = torch.zeros(height * width, 2)
        for i in range(height):
            for j in range(width):
                pos_enc[i * width + j] = torch.tensor([i / height, j / width])
        
        x = torch.cat([x, pos_enc.to(x.device)], dim=1)
        
        return x, edge_index
    

class PretrainedGCNKNN(nn.Module):
    def __init__(self, device, pretrained_output_dim=2048, gcn_hidden_dim=2048, gcn_output_dim=128):
        super(PretrainedGCNKNN, self).__init__()
        self.device = device

        # Pretrained ResNet152
        self.model = models.resnet152(weights='IMAGENET1K_V2')
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the last FC layer
        self.model = self.model.to(device)

        # Unfreeze some layers of ResNet
        for param in list(self.model.parameters())[-10:]:
            param.requires_grad = True

        # GCN layers
        self.gcn1 = GCNConv(pretrained_output_dim, gcn_hidden_dim).to(device)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim).to(device)
        self.gcn3 = GCNConv(gcn_hidden_dim, gcn_output_dim).to(device)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(gcn_hidden_dim).to(device)
        self.bn2 = nn.BatchNorm1d(gcn_hidden_dim).to(device)
        self.bn3 = nn.BatchNorm1d(gcn_output_dim).to(device)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Linear layers for skip connections
        self.skip_lin1 = nn.Linear(gcn_hidden_dim, pretrained_output_dim).to(device)
        self.skip_lin2 = nn.Linear(gcn_hidden_dim, pretrained_output_dim).to(device)
        self.skip_lin3 = nn.Linear(gcn_output_dim, pretrained_output_dim).to(device)

        self.shape = pretrained_output_dim

    def custom_knn_graph(self, x, k=10):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, num_features = x.size(0), x.size(1)
        
        dist = torch.cdist(x, x)
        k = min(k, batch_size - 1)
        _, idx = dist.topk(k + 1, largest=False, dim=-1)
        idx = idx[:, 1:].contiguous() 

        row = torch.arange(batch_size, device=x.device).view(-1, 1).repeat(1, k).view(-1)
        col = idx.view(-1)

        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    @staticmethod
    def preprocess_image(image, device, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform(image).to(device)

    def forward(self, images):
        return self.extract_features(images)

    def extract_features(self, images):
        self.eval()  # Chuyển mô hình sang chế độ eval

        with torch.no_grad():
            features = self.model(images)
            features = features.squeeze()

        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        edge_index = self.custom_knn_graph(features, k=10).to(self.device)

        # Apply GCN with skip connections and linear layers for dimension matching
        x1 = F.relu(self.bn1(self.gcn1(features, edge_index)))
        x1_skip = self.skip_lin1(x1) # Đưa x1 qua skip_lin1 trước khi cộng
        x1 = x1_skip + features  # Ánh xạ và cộng với features
        x1 = self.dropout(x1)

        x2 = F.relu(self.bn2(self.gcn2(x1, edge_index)))
        x2_skip = self.skip_lin2(x2) # Đưa x2 qua skip_lin2 trước khi cộng
        x2 = x2_skip + features  # Ánh xạ và cộng với features
        x2 = self.dropout(x2)

        x3 = self.bn3(self.gcn3(x2, edge_index))
        x3_skip = self.skip_lin3(x3) # Đưa x3 qua skip_lin3 trước khi cộng
        x3 = x3_skip + features  # Ánh xạ và cộng với features

        # Final normalization
        x3 = F.normalize(x3, p=2, dim=1)

        return x3.cpu().detach().numpy()