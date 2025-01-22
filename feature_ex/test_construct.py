import torch
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
from torch import nn
import numpy as np
# 加载EfficientNet模型
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b7').to(device)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 18)
model.load_state_dict(torch.load('../head_5zhe_data/head_data_1/weights-b7-head/EfficientNet_50_18/best_network_eb4.pth', map_location=device))
model.eval()
#print(model)

# 加载图像并进行预处理
image_path = '/home/cz/data/final_data/head_5zhe_data/head_data_1/head_18_r_p/test/Condylura/Condylura#cristata#AMNH14878#DSC_4891#s#v.JPG'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

# 提取每个阶段的特征
endpoints = model.extract_endpoints(input_image)
# print(endpoints['reduction_1'].shape)
# print(endpoints['reduction_2'].shape)
# print(endpoints['reduction_3'].shape)
# print(endpoints['reduction_4'].shape)
# print(endpoints['reduction_5'].shape)
# print(endpoints['reduction_6'].shape)
