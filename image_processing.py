import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Load MiDaS model for depth estimation
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
transform = T.Compose([
    T.Resize(384),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def estimate_depth(image_path):
    img = Image.open(image_path)
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        depth_map = model(input_tensor)
    
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return depth_map


def generate_point_cloud(img_path, depth_map):
    img = cv2.imread(img_path)
    h, w = depth_map.shape
    f = max(h, w)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    pixels = np.stack([j, i, np.ones_like(i)], axis=-1)
    points = pixels @ np.linalg.inv(K).T * depth_map[..., None]
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return points.reshape(-1, 3), colors.reshape(-1, 3)


def noise_reduction(image_path):
    img = cv2.imread(image_path)
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return denoised_img


def color_processing(image_path):
    img = cv2.imread(image_path)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return processed_img
