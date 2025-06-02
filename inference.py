import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from dataloader import color_cluster
from mymodels import Color2Sketch, Sketch2Color

def load_and_transform_image(image_path, transform):
    """
    Read and convert image to tensor.
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor

def load_model(checkpoint_path, model_type="both"):
    """
    Load the model from the checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model_type (str): "both" (Color2Sketch and Sketch2Color), "color2sketch", or "sketch2color".

    Returns:
        tuple or model: The loaded models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "both":
        color2sketch = Color2Sketch(pretrained=False).to(device)
        sketch2color = Sketch2Color(pretrained=False).to(device)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            color2sketch.load_state_dict(checkpoint['color2sketch'], strict=True)
            sketch2color.load_state_dict(checkpoint['sketch2color'], strict=True)
            st.success(f"Loaded checkpoint from {checkpoint_path}")
        color2sketch.eval()
        sketch2color.eval()
        return color2sketch, sketch2color
    elif model_type == "sketch2color":
        sketch2color = Sketch2Color(pretrained=False).to(device)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            sketch2color.load_state_dict(checkpoint['sketch2color'], strict=True)
            st.success(f"Loaded Sketch2Color from {checkpoint_path}")
        sketch2color.eval()
        return sketch2color

def color_to_sketch_and_recolor(checkpoint_path, color_image, selected_style=0, nclusters=9):
    """
    Extract the color palette from the color image, convert it to a sketch image, and recolor it.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        color_image (PIL.Image): Input color image.
        selected_style (int): Style index (0-3).
        nclusters (int): Maximum number of colors in the palette.

    Returns:
        tuple: (fake_color_np, sketch_image_np, color_image_np, palette_preview) - Display result.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    color2sketch, sketch2color = load_model(checkpoint_path, model_type="both")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    color_image_tensor = transform(color_image).unsqueeze(0).to(device)
    color_image_np = color_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    color_image_np_uint8 = (color_image_np * 255).astype(np.uint8)
    palette_images = color_cluster(color_image_np_uint8, nclusters=nclusters)
    palette_tensors = []
    for palette_img in palette_images:
        palette_tensor = transform(Image.fromarray(palette_img)).to(device)
        palette_tensors.append(palette_tensor)
    palette_tensors = torch.stack(palette_tensors).unsqueeze(0)
    with torch.no_grad():
        sketch_image = color2sketch(color_image_tensor)
        num_styles = 4
        style_codes = torch.zeros(1, num_styles).to(device)
        style_codes[0, selected_style] = 1.0
        fake_color = sketch2color(sketch_image, style_codes)
    sketch_image_np = sketch_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    fake_color_np = fake_color.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    palette_preview = np.hstack([cv2.resize(p, (32, 32)) for p in palette_images]) / 255.0
    return fake_color_np, sketch_image_np, color_image_np, palette_preview

def sketch_to_color(checkpoint_path, color_image, sketch_image, selected_style=0, nclusters=9):
    """
    Colorize sketch image based on color palette from color image.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        color_image (PIL.Image): Color image to extract color palette.
        sketch_image (PIL.Image): Input sketch image.
        selected_style (int): Style index (0-3).
        nclusters (int): Maximum number of colors in palette.

    Returns:
        tuple: (fake_color_np, sketch_image_np, color_image_np, palette_preview) - Display result.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sketch2color = load_model(checkpoint_path, model_type="sketch2color")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    color_image_tensor = transform(color_image).unsqueeze(0).to(device)
    sketch_image_tensor = transform(sketch_image).unsqueeze(0).to(device)
    color_image_np = color_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    color_image_np_uint8 = (color_image_np * 255).astype(np.uint8)
    palette_images = color_cluster(color_image_np_uint8, nclusters=nclusters)
    palette_tensors = []
    for palette_img in palette_images:
        palette_tensor = transform(Image.fromarray(palette_img)).to(device)
        palette_tensors.append(palette_tensor)
    palette_tensors = torch.stack(palette_tensors).unsqueeze(0)
    num_styles = 4
    style_codes = torch.zeros(1, num_styles).to(device)
    style_codes[0, selected_style] = 1.0
    with torch.no_grad():
        fake_color = sketch2color(sketch_image_tensor, style_codes)
    sketch_image_np = sketch_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    fake_color_np = fake_color.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    palette_preview = np.hstack([cv2.resize(p, (32, 32)) for p in palette_images]) / 255.0
    return fake_color_np, sketch_image_np, color_image_np, palette_preview

# Streamlit app
st.title("Sketch-to-Color GAN Inference")

st.write("""
This application allows you to:
- Convert a color image to a sketch and recolor it (`color_to_sketch_and_recolor`).
- Colorize a provided sketch using a color palette from a color image (`sketch_to_color`).
""")

# Select function
function_choice = st.selectbox("Select Function", ["color_to_sketch_and_recolor", "sketch_to_color"])

# Upload checkpoint
checkpoint_file = st.file_uploader("Upload Checkpoint File (*.pth)", type=["pth"])
checkpoint_path = None
if checkpoint_file is not None:
    with open("temp_checkpoint.pth", "wb") as f:
        f.write(checkpoint_file.getbuffer())
    checkpoint_path = "temp_checkpoint.pth"

# Upload color image
color_image_file = st.file_uploader("Upload Color Image", type=["png", "jpg", "jpeg"])
color_image = None
if color_image_file is not None:
    color_image = Image.open(color_image_file).convert('RGB')

# Upload sketch image (for sketch_to_color)
sketch_image_file = None
sketch_image = None
if function_choice == "sketch_to_color":
    sketch_image_file = st.file_uploader("Upload Sketch Image", type=["png", "jpg", "jpeg"])
    if sketch_image_file is not None:
        sketch_image = Image.open(sketch_image_file).convert('RGB')

# Select style and nclusters
selected_style = st.slider("Select Style (ghibli-madhouse-sunrise-toei) - (0-3)", 0, 3, 0)

# Run inference
if st.button("Run Inference"):
    if checkpoint_path is None:
        st.error("Please upload a checkpoint file.")
    elif color_image is None:
        st.error("Please upload a color image.")
    elif function_choice == "sketch_to_color" and sketch_image is None:
        st.error("Please upload a sketch image for sketch_to_color.")
    else:
        try:
            if function_choice == "color_to_sketch_and_recolor":
                fake_color_np, sketch_image_np, color_image_np, palette_preview = color_to_sketch_and_recolor(
                    checkpoint_path, color_image, selected_style
                )
                st.subheader("Results")
                cols = st.columns(4)
                with cols[0]:
                    st.image(color_image_np, caption="Original Color", use_column_width=True)
                with cols[1]:
                    st.image(fake_color_np, caption=f"Recolored (Style {selected_style + 1})", use_column_width=True)
                with cols[2]:
                    st.image(palette_preview, caption="Color Palette", use_column_width=True)
            else:  # sketch_to_color
                fake_color_np, sketch_image_np, color_image_np, palette_preview = sketch_to_color(
                    checkpoint_path, color_image, sketch_image, selected_style
                )
                st.subheader("Results")
                cols = st.columns(4)
                with cols[0]:
                    st.image(color_image_np, caption="Original Color", use_column_width=True)
                with cols[1]:
                    st.image(sketch_image_np, caption="Sketch Input", use_column_width=True)
                with cols[2]:
                    st.image(fake_color_np, caption=f"Generated (Style {selected_style + 1})", use_column_width=True)
                with cols[3]:
                    st.image(palette_preview, caption="Color Palette", use_column_width=True)
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")