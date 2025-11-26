import sys
import os
sys.path.append(os.path.abspath('.'))

import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F

import gradio as gr
from gradio import ImageSlider
from huggingface_hub import hf_hub_download
import uuid
from PIL import Image
from datetime import datetime
from glob import glob
import subprocess
import platform
import shutil
import json

model_size = 'large'
MAX_GALLERY_IMAGES = 50

# Create outputs folder
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Test images folder
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'test_images')

# Gradio temp directory (from pinokio config)
GRADIO_TEMP_DIR = os.environ.get('GRADIO_TEMP_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'GRADIO_TEMP_DIR'))

# Settings file for persisting preferences
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

def load_settings():
    """Load settings from file"""
    defaults = {'clear_temp_on_start': False, 'autosave': True}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return {**defaults, **json.load(f)}
        except:
            pass
    return defaults

def save_settings(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")

def clear_temp_directory():
    """Clear the Gradio temp directory"""
    if os.path.exists(GRADIO_TEMP_DIR):
        try:
            for item in os.listdir(GRADIO_TEMP_DIR):
                item_path = os.path.join(GRADIO_TEMP_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print(f"Cleared temp directory: {GRADIO_TEMP_DIR}")
            return True
        except Exception as e:
            print(f"Error clearing temp dir: {e}")
            return False
    return True

# Load settings and clear temp if enabled
settings = load_settings()
if settings.get('clear_temp_on_start', False):
    clear_temp_directory()

# Automatically download model
def get_ddcolor_model_path():
    print("Checking for DDColor model...")
    repo_id = "piddnad/ddcolor_modelscope"
    filename = "pytorch_model.bin"
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Model loaded from: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

class ImageColorizationPipeline(object):
    def __init__(self, model_path, input_size=256, model_size='large'):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.model = DDColor(
            encoder_name=self.encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[self.input_size, self.input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(self.device)

        # Fix for mismatched state dict keys
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
    
    def update_input_size(self, new_size):
        """Rebuild model with new input size"""
        if new_size == self.input_size:
            return False
        self.input_size = new_size
        self.model = DDColor(
            encoder_name=self.encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[self.input_size, self.input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(self.device)
        
        checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        return True

    @torch.no_grad()
    def process(self, img):
        if img is None: return None
        self.height, self.width = img.shape[:2]
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()

        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        return output_img

# --- INIT ---
model_file_path = get_ddcolor_model_path()
if model_file_path is None:
    raise RuntimeError("Could not download or find the DDColor model.")

colorizer = ImageColorizationPipeline(model_path=model_file_path,
                                      input_size=512,
                                      model_size=model_size)

# Store last result for manual save and status update
last_colorized_image = None
last_status_message = ""
last_status_type = "info"

def status_html(message, status_type="info"):
    """Generate styled HTML status message"""
    if status_type == "success":
        return f'<div style="padding: 8px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724;">{message}</div>'
    elif status_type == "error":
        return f'<div style="padding: 8px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24;">{message}</div>'
    else:  # info
        return f'<div style="padding: 8px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; color: #0c5460;">{message}</div>'

def get_saved_images():
    """Get list of saved output images, newest first, limited"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(OUTPUTS_DIR, ext)))
    # Sort by modification time, newest first
    images.sort(key=os.path.getmtime, reverse=True)
    return images[:MAX_GALLERY_IMAGES]

def colorize(img, autosave, input_size):
    global last_colorized_image, last_status_message, last_status_type
    
    if img is None:
        last_status_message = "Please upload an image first."
        last_status_type = "error"
        return None
    
    # Update model input size if changed
    model_reloaded = colorizer.update_input_size(input_size)
    
    image_out = colorizer.process(img)
    last_colorized_image = image_out.copy()  # Store for manual save
    
    # Convert BGR to RGB for Gradio display
    image_out_rgb = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    
    # Build status message
    reload_note = " (model reloaded)" if model_reloaded else ""
    
    # Autosave if enabled
    if autosave:
        saved_path = save_output_image(image_out)
        last_status_message = f"✅ Auto-saved: {os.path.basename(saved_path)}{reload_note}"
        last_status_type = "success"
    else:
        last_status_message = f"Colorized at {input_size}px{reload_note}"
        last_status_type = "info"
    
    # Return numpy arrays directly - Gradio handles temp files internally
    return (img, image_out_rgb)

def save_output_image(image_data=None):
    """Save image to outputs folder"""
    if image_data is None:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"colorized_{timestamp}.png"
    save_path = os.path.join(OUTPUTS_DIR, filename)
    cv2.imwrite(save_path, image_data)
    return save_path

def update_status_and_gallery():
    """Update status and gallery after colorization completes"""
    return status_html(last_status_message, last_status_type), get_saved_images()

def manual_save():
    """Manually save the current output using stored image"""
    global last_colorized_image
    
    if last_colorized_image is None:
        return status_html("No image to save. Run colorization first.", "error"), get_saved_images()
    
    saved_path = save_output_image(last_colorized_image)
    return status_html(f"✅ Saved: {os.path.basename(saved_path)}", "success"), get_saved_images()

def open_outputs_folder():
    try:
        if platform.system() == "Windows":
            os.startfile(OUTPUTS_DIR)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", OUTPUTS_DIR])
        else:  # Linux
            subprocess.run(["xdg-open", OUTPUTS_DIR])
        return status_html("📂 Opened outputs folder", "info")
    except Exception as e:
        return status_html(f"Could not open folder: {e}", "error")

def clear_all():
    global last_colorized_image
    last_colorized_image = None
    return None, None, ""

def get_test_images():
    """Get list of test images for gallery"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    images = []
    for ext in extensions:
        images.extend(glob(os.path.join(TEST_IMAGES_DIR, ext)))
    return images

def load_test_image(evt: gr.SelectData):
    """Load selected test image into the input"""
    test_images = get_test_images()
    if evt.index < len(test_images):
        return test_images[evt.index]
    return None

# CSS for responsive image windows
css = """
.image-window {
    min-height: 300px !important;
    height: auto !important;
}
.image-window img {
    max-height: 60vh !important;
    object-fit: contain;
    width: 100%;
}
"""

# Gradio demo using the Image-Slider custom component
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            bw_image = gr.Image(label='Black and White Input Image', elem_classes="image-window")
            
            with gr.Row():
                btn = gr.Button('Convert using DDColor', variant='primary', size="sm")
                clear_btn = gr.Button('Clear', variant='secondary', scale=0, size="sm")

            input_size_radio = gr.Radio(
                choices=[512, 768, 1024],
                value=512,
                label="Processing Size",
                info="Higher = potentially better quality, more VRAM (~3GB/7GB/9GB). Changing size will reload the model - 1st gen always slow."
            )

            # Test images accordion
            with gr.Accordion("📁 Sample Test Images", open=False):
                gr.Markdown("*Click an image to load it as input*")
                test_gallery = gr.Gallery(
                    value=get_test_images(),
                    label="Test Images",
                    columns=4,
                    height=300,
                    object_fit="contain",
                    allow_preview=False
                )
                
        with gr.Column():
            col_image_slider = gr.ImageSlider(
                label="Before/After Comparison",
                interactive=False,
                elem_classes="image-window"
            )
            
            # output controls
            with gr.Group():
                with gr.Row():
                    autosave_checkbox = gr.Checkbox(label="Auto-save outputs", value=settings.get('autosave', True))
                    clear_temp_checkbox = gr.Checkbox(
                        label="Clear temp on start", 
                        value=settings.get('clear_temp_on_start', False),
                    )
                with gr.Row():                                
                    save_btn = gr.Button('💾 Save', variant='secondary', scale=0, size="sm")
                    open_folder_btn = gr.Button('📂 Open Folder', variant='secondary', scale=0, size="sm")
                
                save_status = gr.HTML()

    with gr.Row():            
        # Recent outputs gallery
        with gr.Accordion("🖼️ Recent Outputs", open=False):
            output_gallery = gr.Gallery(
                value=get_saved_images(),
                label="Saved Images",
                columns=5,
                height=800,
                object_fit="contain",
                allow_preview=True
            )

    def on_clear_temp_change(value):
        """Save clear temp preference when changed"""
        settings['clear_temp_on_start'] = value
        save_settings(settings)
    
    def on_autosave_change(value):
        """Save autosave preference when changed"""
        settings['autosave'] = value
        save_settings(settings)
    
    # Event handlers - chain status/gallery update after colorize to avoid processing overlay
    btn.click(
        colorize, 
        [bw_image, autosave_checkbox, input_size_radio], 
        [col_image_slider]
    ).then(
        update_status_and_gallery,
        None,
        [save_status, output_gallery]
    )
    
    save_btn.click(manual_save, None, [save_status, output_gallery])
    open_folder_btn.click(open_outputs_folder, None, save_status)
    clear_btn.click(clear_all, None, [bw_image, col_image_slider, save_status])
    clear_temp_checkbox.change(on_clear_temp_change, [clear_temp_checkbox], None)
    autosave_checkbox.change(on_autosave_change, [autosave_checkbox], None)
    test_gallery.select(load_test_image, None, bw_image)

demo.launch(css=css)
