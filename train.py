import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataloader import PairImageFolder
from mymodels import Color2Sketch, Sketch2Color, Discriminator
from losses import LossCollector
import os
import numpy as np
import random
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import torch.multiprocessing as mp

def train(args):
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Tạo transform nếu chưa được định nghĩa
    if not hasattr(args, 'transform'):
        args.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    color2sketch = Color2Sketch().to(device)
    sketch2color = Sketch2Color().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize loss collector
    loss_collector = LossCollector().to(device)
    
    # Initialize optimizers
    optimizer_G = optim.Adam(
        list(color2sketch.parameters()) + list(sketch2color.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=args.lr, betas=(0.5, 0.999)
    )
    
    # Initialize dataloaders
    train_dataset = PairImageFolder(
        root=args.data_dir,
        transform=args.transform,
        sketch_net=color2sketch,
        nclusters=args.nclusters
    )

    # Tính toán kích thước train và test
    train_size = int(0.8 * len(train_dataset))  # 80% cho training
    test_size = len(train_dataset) - train_size
    
    # Chia dataset
    train_dataset, test_dataset = random_split(
        train_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Đảm bảo tính tái lập
    )

    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    def evaluate():
        color2sketch.eval()
        sketch2color.eval()
        discriminator.eval()
        
        test_losses = {
            'total': 0,
            'adv': 0,
            'feat': 0,
            'perc': 0,
            'style': 0,
            'color': 0  # Thêm color loss vào test losses
        }

        with torch.no_grad():
            for batch in test_loader:
                real_sketch, real_color, color_palette = batch
                real_sketch = real_sketch.to(device)
                real_color = real_color.to(device)
                color_palette = color_palette.to(device)  # Chuyển bảng màu sang device
                
                # Generate style codes
                batch_size = real_sketch.size(0)
                if args.random_style:
                    # Tạo style codes ngẫu nhiên
                    style_codes = torch.randn(batch_size, 4, requires_grad=False).to(device)
                    style_codes = F.softmax(style_codes, dim=1)
                else:
                    # Sử dụng one-hot encoding
                    style_codes = torch.eye(4, requires_grad=False).to(device)
                    style_codes = style_codes.repeat(batch_size // 4 + 1, 1)[:batch_size]
                
                # Forward pass
                fake_sketch = color2sketch(real_color, style_codes)
                fake_color = sketch2color(real_sketch, style_codes)
                
                # Get features
                real_features = discriminator(real_color, real_sketch)
                fake_features = discriminator(fake_color, fake_sketch)
                
                # Calculate losses
                losses = loss_collector(
                    real_color, fake_color,
                    real_features, fake_features,
                    color_palette  # Truyền bảng màu vào để tính toán color loss
                )
                
                # Accumulate losses
                for k, v in losses.items():
                    test_losses[k] += v.item()
        
        # Average losses
        for k in test_losses:
            test_losses[k] /= len(test_loader)
            
        return test_losses

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))  # Add map_location
            
            # Load model weights
            color2sketch.load_state_dict(checkpoint['color2sketch'])
            sketch2color.load_state_dict(checkpoint['sketch2color'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            
            # Load optimizer states
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            
            # Load training state
            start_epoch = checkpoint['epoch']
            
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        color2sketch.train()
        sketch2color.train()
        discriminator.train()

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')

        for batch in progress_bar:
            real_sketch, real_color, color_palette = batch  # Nhận bảng màu từ batch
            real_sketch = real_sketch.to(device)
            real_color = real_color.to(device)
            color_palette = color_palette.to(device)  # Chuyển bảng màu sang device

            # Generate style codes
            batch_size = real_sketch.size(0)  # Lấy kích thước batch từ tensor
            if args.random_style:
                style_codes = torch.randn(batch_size, 4, requires_grad=True).to(device)
                style_codes = F.softmax(style_codes, dim=1)  # Chuẩn hóa thành phân phối xác suất
            else:
                # Sử dụng one-hot encoding
                style_codes = torch.eye(4, requires_grad=True).to(device)
                style_codes = style_codes.repeat(batch_size // 4 + 1, 1)[:batch_size]

            # Train Generator
            optimizer_G.zero_grad()

            # Color to Sketch
            fake_sketch = color2sketch(real_color, style_codes)

            # Sketch to Color
            fake_color = sketch2color(fake_sketch, style_codes)

            # Tính toán loss dựa trên bảng màu
            color_loss = loss_collector.color_loss(fake_color, color_palette)  # Hàm tính toán loss dựa trên bảng màu

            # Get features from discriminator
            real_features = discriminator(real_color, real_sketch)
            fake_features = discriminator(fake_color, fake_sketch)

            # Calculate losses
            losses = loss_collector(
                real_color, fake_color,
                real_features, fake_features,
                color_palette  # Truyền bảng màu vào để tính toán color loss
            )

            # Kết hợp color_loss vào tổng loss
            losses['total'].backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real samples
            real_pred = discriminator(real_color, real_sketch)
            real_loss = loss_collector.adversarial_loss(real_pred, torch.ones_like(real_pred))

            # Fake samples
            fake_pred = discriminator(fake_color.detach(), fake_sketch.detach())
            fake_loss = loss_collector.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': losses['total'].item(),
                'D_loss': d_loss.item(),
                'Style': losses['style'].item()
            })

        # Evaluation phase
        if (epoch + 1) % args.eval_interval == 0:
            test_losses = evaluate()
            print(f"\nEpoch {epoch + 1} Test Losses:")
            for k, v in test_losses.items():
                print(f"{k}: {v:.4f}")

        # Save checkpoints
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'color2sketch': color2sketch.state_dict(),
                'sketch2color': sketch2color.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, f'checkpoint_epoch_{epoch + 1}.pth')

def save_checkpoint(state, filename):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(state, os.path.join('checkpoints', filename))

if __name__ == '__main__':
    # Set start method to 'spawn'
    mp.set_start_method('spawn', force=True)
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--nclusters', type=int, default=9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=5, help='Number of epochs between evaluations')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
    parser.add_argument('--random_style', action='store_true', help='Use random style codes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin memory in DataLoader')

    args = parser.parse_args()
    train(args) 