import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # vgg = models.vgg19(pretrained=True).features
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h
        out = [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]
        return out

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(h * w)

class LossCollector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19()
        self.gram = GramMatrix()
        
        # Loss weights
        self.lambda_adv = 1.0
        self.lambda_feat = 10.0
        self.lambda_perc = 10.0
        self.lambda_style = 5.0
        self.lambda_color = 1.0  # Thêm trọng số cho color loss
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def adversarial_loss(self, pred, target):
        return self.mse_loss(pred, target)
        
    def feature_matching_loss(self, real_features, fake_features):
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += self.l1_loss(fake_feat, real_feat)
        return loss
        
    def perceptual_loss(self, real_img, fake_img):
        real_features = self.vgg(real_img)
        fake_features = self.vgg(fake_img)
        return self.feature_matching_loss(real_features, fake_features)
        
    def style_loss(self, real_img, fake_img):
        real_features = self.vgg(real_img)
        fake_features = self.vgg(fake_img)
        
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            real_gram = self.gram(real_feat)
            fake_gram = self.gram(fake_feat)
            loss += self.l1_loss(fake_gram, real_gram)
        return loss

    def color_loss(self, fake_img, color_palette):
        # Giả sử color_palette có kích thước (batch_size, n_colors, C, H, W)
        batch_size = fake_img.size(0)
        n_colors = color_palette.size(1)  # Số lượng màu trong bảng màu

        # Tính toán loss cho từng màu trong bảng màu
        loss = 0
        for i in range(batch_size):
            for j in range(n_colors):
                # Tính toán loss cho từng màu
                loss += self.l1_loss(fake_img[i], color_palette[i, j])  # color_palette[i, j] có kích thước (3, 256, 256)

        return loss / (batch_size * n_colors)  # Trung bình loss

    def forward(self, real_img, fake_img, real_features, fake_features, color_palette):
        # Adversarial loss
        adv_loss = self.adversarial_loss(real_img, fake_img)
        
        # Feature matching loss
        feat_loss = self.feature_matching_loss(real_features, fake_features)
        
        # Perceptual loss
        perc_loss = self.perceptual_loss(real_img, fake_img)
        
        # Style loss
        style_loss = self.style_loss(real_img, fake_img)
        
        # Color loss
        col_loss = self.color_loss(fake_img, color_palette) 
        
        # Total loss
        total_loss = (self.lambda_adv * adv_loss + 
                      self.lambda_feat * feat_loss + 
                      self.lambda_perc * perc_loss + 
                      self.lambda_style * style_loss +
                      self.lambda_color * col_loss) 
        
        return {
            'total': total_loss,
            'adv': adv_loss,
            'feat': feat_loss,
            'perc': perc_loss,
            'style': style_loss,
            'color': col_loss 
        }