import torch
import torch.nn as nn
import os

__all__ = [
  'Color2Sketch', 'Sketch2Color', 'Discriminator', 
]

# Thêm nhiễu vào ảnh
class ApplyNoise(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(channels))

  def forward(self, x, noise=None):
    # Nếu None thì tạo ngẫu nhiên, nếu được cung cấp thì sử dụng luôn
    if noise is None:
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
    return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)

# Giống Conv nhưng chuẩn hóa trước khi tích chập
class Conv2d_WS(nn.Conv2d):
  def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_chan, out_chan, kernel_size, stride, padding, dilation, groups, bias)
  
  def forward(self, x):
    weight = self.weight
    weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,keepdim=True).mean(dim=3, keepdim=True)
    weight = weight - weight_mean
    std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1,1)+1e-5
    weight = weight / std.expand_as(weight)
    return torch.nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# Residual Block kết hợp Group Normalization và Weight Standardization 
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, sample=None):
    super(ResidualBlock, self).__init__()
    self.ic = in_channels
    self.oc = out_channels
    self.conv1 = Conv2d_WS(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.GroupNorm(32, out_channels)
    self.conv2 = Conv2d_WS(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.GroupNorm(32, out_channels)
    self.convr = Conv2d_WS(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    self.bnr = nn.GroupNorm(32, out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.sample = sample
    if self.sample == 'down':
        self.sampling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    elif self.sample == 'up':
        self.sampling = nn.Upsample(scale_factor=2, mode='nearest')
      
  def forward(self, x):
    if self.ic != self.oc:
        residual = self.convr(x)
        residual = self.bnr(residual)
    else:
        residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = self.relu(out)
    if self.sample is not None:
        out = self.sampling(out)
    return out

class MappingNetwork(nn.Module):
    def __init__(self, style_dim=4, hidden_dim=512, out_dim=512):
        super().__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(style_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, out_dim)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, style_code):
        # Convert one-hot style code to embedding
        x = self.leaky_relu(self.fc1(style_code))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.leaky_relu(self.fc5(x))
        x = self.leaky_relu(self.fc6(x))
        x = self.leaky_relu(self.fc7(x))
        x = self.fc8(x)
        return x

class AdaIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, x, w):
        # Split w into scale and bias
        scale = w[:, :self.num_features].view(-1, self.num_features, 1, 1)
        bias = w[:, self.num_features:].view(-1, self.num_features, 1, 1)
        
        # Normalize input
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std
        
        # Apply style
        x = x * scale + bias
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            Conv2d_WS(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int)
        )
        
        self.W_x = nn.Sequential(
            Conv2d_WS(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32, F_int)
        )

        self.psi = nn.Sequential(
            Conv2d_WS(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Style embedding processing
        self.style_fc = nn.Linear(512, F_int * 2)  # Split into scale and bias
        self.adain = AdaIN(F_int)
        
    def forward(self, g, x, w=None):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if w is not None:
            # Process style embedding
            style_params = self.style_fc(w)
            g1 = self.adain(g1, style_params)
            x1 = self.adain(x1, style_params)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Color2SketchEncoder(nn.Module):
    def __init__(self, nc=3):
        super(Color2SketchEncoder, self).__init__()
        self.layer1 = ResidualBlock(nc, 64, sample='down')
        self.layer2 = ResidualBlock(64, 128, sample='down')
        self.layer3 = ResidualBlock(128, 256, sample='down')
        self.layer4 = ResidualBlock(256, 512, sample='down')
        self.layer5 = ResidualBlock(512, 512, sample='down')
        self.layer6 = ResidualBlock(512, 512, sample='down')
        self.layer7 = ResidualBlock(512, 512, sample='down')

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        return x1, x2, x3, x4, x5, x6, x7

class Color2SketchDecoder(nn.Module):
            def __init__(self, mapping_network):
                super(Color2SketchDecoder, self).__init__()
                self.mapping_network = mapping_network
                # Convolutional layers và upsampling     
                self.noise7 = ApplyNoise(512)
                self.layer7_up = ResidualBlock(512, 512, sample='up')
                
                self.Att6 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = ApplyNoise(512)
                self.layer6_up = ResidualBlock(512, 512, sample='up')
                
                self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = ApplyNoise(512)
                self.layer5_up = ResidualBlock(512, 512, sample='up')
                
                self.Att4 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = ApplyNoise(512)
                self.layer4_up = ResidualBlock(512, 256, sample='up')
                
                self.Att3 = Attention_block(F_g=256,F_l=256,F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = ApplyNoise(256)
                self.layer3_up = ResidualBlock(256, 128, sample='up')
                
                self.Att2 = Attention_block(F_g=128,F_l=128,F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = ApplyNoise(128)
                self.layer2_up = ResidualBlock(128, 64, sample='up')
                
                self.Att1 = Attention_block(F_g=64,F_l=64,F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = ApplyNoise(64)
                self.layer1_up = ResidualBlock(64, 32, sample='up')   
                
                self.noise0 = ApplyNoise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.activation = nn.ReLU(inplace=True)
                self.tanh = nn.Tanh()

            def forward(self, midlevel_input, style_code=None):
                x1, x2, x3, x4, x5, x6, x7 = midlevel_input
                
                # Get style embedding if provided
                w = None
                if style_code is not None:
                    w = self.mapping_network(style_code)
                
                x = self.noise7(x7)                
                x = self.layer7_up(x)

                x6 = self.Att6(g=x, x=x6, w=w)
                x = torch.cat((x, x6), dim=1)
                x = self.layer6(x)
                x = self.noise6(x)
                x = self.layer6_up(x)
                
                x5 = self.Att5(g=x, x=x5, w=w)
                x = torch.cat((x, x5), dim=1)
                x = self.layer5(x)
                x = self.noise5(x)
                x = self.layer5_up(x)

                x4 = self.Att4(g=x, x=x4, w=w)
                x = torch.cat((x, x4), dim=1)
                x = self.layer4(x)
                x = self.noise4(x)
                x = self.layer4_up(x)
                
                x3 = self.Att3(g=x, x=x3, w=w)
                x = torch.cat((x, x3), dim=1)
                x = self.layer3(x)
                x = self.noise3(x)
                x = self.layer3_up(x)
                
                x2 = self.Att2(g=x, x=x2, w=w)
                x = torch.cat((x, x2), dim=1)
                x = self.layer2(x)
                x = self.noise2(x)
                x = self.layer2_up(x)
                
                x1 = self.Att1(g=x, x=x1, w=w)
                x = torch.cat((x, x1), dim=1)
                x = self.layer1(x)
                x = self.noise1(x)
                x = self.layer1_up(x)
                
                x = self.noise0(x)
                x = self.layer0(x)
                x = self.tanh(x)

                return x

class Color2Sketch(nn.Module):
    def __init__(self, nc=3, pretrained=False):
        super(Color2Sketch, self).__init__()
        self.mapping_network = MappingNetwork()
        self.encoder = Color2SketchEncoder(nc)
        self.decoder = Color2SketchDecoder(self.mapping_network)
        if pretrained:
            print('Loading pretrained {0} model...'.format('Color2Sketch'), end=' ')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/color2edge/ckpt.pth')
            self.load_state_dict(checkpoint['netG'], strict=True)
            print("Done!")
        else:
            self.apply(weights_init)
            print('Weights of {0} model are initialized'.format('Color2Sketch'))
        
    def forward(self, inputs, style_code=None):
        encode = self.encoder(inputs)
        output = self.decoder(encode, style_code)
        return output

class Sketch2ColorEncoder(nn.Module):
            def __init__(self, nc=3):
                super(Sketch2ColorEncoder, self).__init__()
                # Build ResNet and change first conv layer to accept single-channel input
                self.layer1 = ResidualBlock(nc, 64, sample='down')
                self.layer2 = ResidualBlock(64, 128, sample='down')
                self.layer3 = ResidualBlock(128, 256, sample='down')
                self.layer4 = ResidualBlock(256, 512, sample='down')
                self.layer5 = ResidualBlock(512, 512, sample='down')
                self.layer6 = ResidualBlock(512, 512, sample='down')
                self.layer7 = ResidualBlock(512, 512, sample='down')

            def forward(self, x):
                # Lưu các feature maps từ mỗi layer
                x1 = self.layer1(x)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
                x5 = self.layer5(x4)
                x6 = self.layer6(x5)
                x7 = self.layer7(x6)
                
                return x1, x2, x3, x4, x5, x6, x7

class Sketch2ColorDecoder(nn.Module):
            def __init__(self, mapping_network):
                super(Sketch2ColorDecoder, self).__init__()
                self.mapping_network = mapping_network
                # Convolutional layers and upsampling     
                self.noise7 = ApplyNoise(512)
                self.layer7_up = ResidualBlock(512, 512, sample='up')
                
                self.Att6 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer6 = ResidualBlock(1024, 512, sample=None)
                self.noise6 = ApplyNoise(512)
                self.layer6_up = ResidualBlock(512, 512, sample='up')
                
                self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer5 = ResidualBlock(1024, 512, sample=None)
                self.noise5 = ApplyNoise(512)
                self.layer5_up = ResidualBlock(512, 512, sample='up')
                
                self.Att4 = Attention_block(F_g=512,F_l=512,F_int=256)
                self.layer4 = ResidualBlock(1024, 512, sample=None)
                self.noise4 = ApplyNoise(512)
                self.layer4_up = ResidualBlock(512, 256, sample='up')
                
                self.Att3 = Attention_block(F_g=256,F_l=256,F_int=128)
                self.layer3 = ResidualBlock(512, 256, sample=None)
                self.noise3 = ApplyNoise(256)
                self.layer3_up = ResidualBlock(256, 128, sample='up')
                
                self.Att2 = Attention_block(F_g=128,F_l=128,F_int=64)
                self.layer2 = ResidualBlock(256, 128, sample=None)
                self.noise2 = ApplyNoise(128)
                self.layer2_up = ResidualBlock(128, 64, sample='up')
                
                self.Att1 = Attention_block(F_g=64,F_l=64,F_int=32)
                self.layer1 = ResidualBlock(128, 64, sample=None)
                self.noise1 = ApplyNoise(64)
                self.layer1_up = ResidualBlock(64, 32, sample='up')   
                
                self.noise0 = ApplyNoise(32)
                self.layer0 = Conv2d_WS(32, 3, kernel_size=3, stride=1, padding=1)
                self.activation = nn.ReLU(inplace=True)
                self.tanh = nn.Tanh()

            def forward(self, midlevel_input, style_code=None):
                x1, x2, x3, x4, x5, x6, x7 = midlevel_input
                
                # Get style embedding if provided
                w = None
                if style_code is not None:
                    w = self.mapping_network(style_code)
                
                x = self.noise7(x7)                
                x = self.layer7_up(x)

                x6 = self.Att6(g=x, x=x6, w=w)
                x = torch.cat((x, x6), dim=1)
                x = self.layer6(x)
                x = self.noise6(x)
                x = self.layer6_up(x)
                
                x5 = self.Att5(g=x, x=x5, w=w)
                x = torch.cat((x, x5), dim=1)
                x = self.layer5(x)
                x = self.noise5(x)
                x = self.layer5_up(x)

                x4 = self.Att4(g=x, x=x4, w=w)
                x = torch.cat((x, x4), dim=1)
                x = self.layer4(x)
                x = self.noise4(x)
                x = self.layer4_up(x)
                
                x3 = self.Att3(g=x, x=x3, w=w)
                x = torch.cat((x, x3), dim=1)
                x = self.layer3(x)
                x = self.noise3(x)
                x = self.layer3_up(x)
                
                x2 = self.Att2(g=x, x=x2, w=w)
                x = torch.cat((x, x2), dim=1)
                x = self.layer2(x)
                x = self.noise2(x)
                x = self.layer2_up(x)
                
                x1 = self.Att1(g=x, x=x1, w=w)
                x = torch.cat((x, x1), dim=1)
                x = self.layer1(x)
                x = self.noise1(x)
                x = self.layer1_up(x)
                
                x = self.noise0(x)
                x = self.layer0(x)
                x = self.tanh(x)

                return x

class Sketch2Color(nn.Module):
    def __init__(self, nc=3, pretrained=False):
        super(Sketch2Color, self).__init__()
        self.mapping_network = MappingNetwork()
        self.encoder = Sketch2ColorEncoder(nc)
        self.decoder = Sketch2ColorDecoder(self.mapping_network)
        if pretrained:
            print('Loading pretrained {0} model...'.format('Sketch2Color'), end=' ')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/edge2color/ckpt.pth')
            self.load_state_dict(checkpoint['netG'], strict=True)
            print("Done!")
        else:
            self.apply(weights_init)
            print('Weights of {0} model are initialized'.format('Sketch2Color'))
            
    def forward(self, inputs, style_code=None):
        encode = self.encoder(inputs)
        output = self.decoder(encode, style_code)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, nc=6, pretrained=False):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.GroupNorm(32, 64)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.GroupNorm(32,128)
        self.conv3 = torch.nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.GroupNorm(32, 256)
        self.conv4 = torch.nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1))
        self.bn4 = nn.GroupNorm(32, 512)
        self.conv5 = torch.nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))               
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        if pretrained:
            pass
        else:
            self.apply(weights_init)
            print('Weights of {0} model are initialized'.format('Discriminator'))

    def forward(self, base, unknown):
        input = torch.cat((base, unknown), dim=1)
        x = self.activation(self.conv1(input))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))

        return x.mean((2,3))

# To initialize model weights
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('Conv2d_WS') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    else:
        pass
