import cv2 
import numpy as np 
import torch 
import torchvision
from torchvision.transforms import functional as FF
from torchvision import datasets
from PIL import Image 

def color_cluster(img, nclusters=9):
    # Chuyển tensor về numpy array nếu cần
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).cpu().numpy()  # Chuyển từ (C,H,W) sang (H,W,C)
        img = (img * 255).astype(np.uint8)  # Chuyển từ [-1,1] sang [0,255]

    img_size = img.shape

    # Thu nhỏ ảnh gốc xuống 25% 
    small_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    # reshape thành mảng 2D (N, 3) với mỗi hàng là một pixel RGB
    sample = small_img.reshape((-1, 3))
    # Convert về float32 để dùng K-Means 
    sample = np.float32(sample)

    # Áp dụng K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(sample, nclusters, None, criteria, 10, flags)

    # Đếm số lượng pixel trong mỗi cụm
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Sắp xếp các cụm theo số lượng pixel giảm dần
    sorted_indices = np.argsort(counts)[::-1]
    sorted_centers = centers[sorted_indices]
    
    # Lấy tối đa 9 màu chủ đạo
    n_colors = min(len(sorted_centers), 9)
    dominant_colors = sorted_centers[:n_colors]

    # Tạo ảnh chỉ chứa màu màu trung tâm
    color_palette = []
    for color in dominant_colors:
        dominant_color = np.zeros(img_size, dtype='uint8')
        dominant_color[:, :, :] = color
        color_palette.append(dominant_color)

    return color_palette

class PairImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, sketch_net, nclusters):
        super(PairImageFolder, self).__init__(root, transform)
        self.nclusters = nclusters 
        self.sketch_net = sketch_net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)  # Đọc ảnh (đã là PIL Image)
        
        # Tạo bản sao numpy để xử lý color clustering
        img_np = np.array(img)
        color_palette = color_cluster(img_np, nclusters=self.nclusters)
        
        # Áp dụng transform cho ảnh gốc
        if self.transform is not None:
            img = self.transform(img)  # Áp dụng biến đổi ảnh
        else:
            img = FF.to_tensor(img)
            img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        with torch.no_grad():
            img_edge = self.sketch_net(img.unsqueeze(0).to(self.device))
            img_edge = img_edge.squeeze().permute(1,2,0).cpu().numpy()
            # Chuyển numpy array thành PIL Image trước khi áp dụng to_grayscale
            img_edge = Image.fromarray((img_edge * 255).astype(np.uint8))
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)

        # Chuyển bảng màu thành tensor và đảm bảo cùng kích thước
        color_palette_tensors = []
        for color in color_palette:
            color_tensor = self.make_tensor(color)
            # Đảm bảo cùng kích thước với ảnh gốc
            if color_tensor.shape != img.shape:
                color_tensor = FF.resize(color_tensor, img.shape[-2:])
            color_palette_tensors.append(color_tensor)

        # Chuyển list thành tensor
        color_palette = torch.stack(color_palette_tensors)

        return img_edge, img, color_palette

    def make_tensor(self, img):
        # Chuyển ảnh NumPy/PIL → PyTorch Tensor
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = FF.to_tensor(img)
        # Chuẩn hóa giá trị pixel từ [0,1] → [-1,1]
        img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return img