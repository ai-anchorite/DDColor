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
from gradio_imageslider import ImageSlider
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
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
VALID_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

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

def is_valid_image(filepath):
    """Check if file is a valid image based on extension"""
    if filepath is None:
        return False
    ext = os.path.splitext(filepath)[1].lower()
    return ext in VALID_IMAGE_EXTENSIONS

def get_images_from_folder(folder_path):
    """Get all valid image files from a folder"""
    if not folder_path or not os.path.isdir(folder_path):
        return []
    images = []
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        if os.path.isfile(full_path) and is_valid_image(full_path):
            images.append(full_path)
    return sorted(images)

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def colorize_images(image_paths, output_dir, progress_tracker=None, desc="Colorizing"):
    """Core colorization function - processes images and saves to output_dir.
    Returns (results_rgb, saved_count, skipped_count)
    """
    global last_colorized_image
    
    results = []
    saved_count = 0
    skipped = 0
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        # Update progress
        if progress_tracker is not None:
            progress_tracker((i + 1) / total, desc=f"{desc} ({i + 1}/{total})")
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            
            output = colorizer.process(img)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            results.append(output_rgb)
            
            # Save to output directory
            original_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{original_name}_colorized.png")
            cv2.imwrite(save_path, output)
            saved_count += 1
            
            last_colorized_image = output.copy()
        except Exception as e:
            skipped += 1
            continue
    
    return results, saved_count, skipped

def process_batch(file_list, folder_path, autosave, input_size, progress=gr.Progress()):
    """Process multiple images from file uploads and/or folder path"""
    global last_status_message, last_status_type
    
    # Update model input size if changed
    colorizer.update_input_size(input_size)
    
    # Collect all valid image paths
    image_paths = []
    
    # From file uploads
    if file_list:
        for f in file_list:
            fpath = f.name if hasattr(f, 'name') else f
            if is_valid_image(fpath):
                image_paths.append(fpath)
    
    # From folder
    if folder_path and folder_path.strip():
        folder_images = get_images_from_folder(folder_path.strip())
        image_paths.extend(folder_images)
    
    if not image_paths:
        last_status_message = "No valid images found. Supported: JPG, PNG, WEBP, BMP, TIFF"
        last_status_type = "error"
        return []
    
    # Create timestamped subfolder for batch output
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder_name = f"batch_{batch_timestamp}"
    batch_output_dir = os.path.join(OUTPUTS_DIR, batch_folder_name)
    os.makedirs(batch_output_dir, exist_ok=True)
    
    results, saved_count, skipped = colorize_images(image_paths, batch_output_dir, progress)
    
    # Build status message
    msg_parts = [f"✅ Processed {len(results)} image(s)"]
    msg_parts.append(f"saved to {batch_folder_name}/")
    if skipped > 0:
        msg_parts.append(f"skipped {skipped}")
    
    last_status_message = ", ".join(msg_parts)
    last_status_type = "success"
    
    return results

def process_video(video_path, input_size, progress=gr.Progress()):
    """Extract frames from video, colorize them, and reassemble"""
    global last_status_message, last_status_type
    
    if not video_path:
        last_status_message = "Please upload a video first."
        last_status_type = "error"
        return None
    
    if not check_ffmpeg():
        last_status_message = "❌ ffmpeg not found. Please install ffmpeg or run via Pinokio."
        last_status_type = "error"
        return None
    
    # Update model input size
    colorizer.update_input_size(input_size)
    
    # Create working directories
    video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = os.path.join(OUTPUTS_DIR, f"video_{video_timestamp}")
    frames_dir = os.path.join(work_dir, "frames")
    colorized_dir = os.path.join(work_dir, "colorized")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(colorized_dir, exist_ok=True)
    
    # Get video info (fps)
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=True
        )
        fps_str = probe.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
    except Exception:
        fps = 30.0  # fallback
    
    # Extract frames
    progress(0.05, desc="Extracting frames (this may take a moment)...")
    try:
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-qscale:v', '2', 
             os.path.join(frames_dir, 'frame_%06d.png')],
            capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        last_status_message = f"❌ Failed to extract frames: {e.stderr.decode()[:200]}"
        last_status_type = "error"
        return None
    
    # Get frame list
    frame_paths = sorted(glob(os.path.join(frames_dir, '*.png')))
    if not frame_paths:
        last_status_message = "❌ No frames extracted from video."
        last_status_type = "error"
        return None
    
    progress(0.1, desc=f"Extracted {len(frame_paths)} frames, starting colorization...")
    
    # Colorize frames
    results, saved_count, skipped = colorize_images(frame_paths, colorized_dir, progress, desc="Colorizing")
    
    if saved_count == 0:
        last_status_message = "❌ Failed to colorize any frames."
        last_status_type = "error"
        return None
    
    # Reassemble video
    progress(0.9, desc="Assembling video...")
    output_video = os.path.join(work_dir, f"{video_name}_colorized.mp4")
    try:
        subprocess.run(
            ['ffmpeg', '-framerate', str(fps), '-i', os.path.join(colorized_dir, 'frame_%06d_colorized.png'),
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', output_video],
            capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        last_status_message = f"❌ Failed to assemble video: {e.stderr.decode()[:200]}"
        last_status_type = "error"
        return None
    
    # Cleanup frame directories to save space
    shutil.rmtree(frames_dir)
    shutil.rmtree(colorized_dir)
    
    last_status_message = f"✅ Video colorized! {saved_count} frames, saved to video_{video_timestamp}/"
    last_status_type = "success"
    
    return output_video

# CSS for responsive media windows
css = """
.media-window {
    min-height: 300px !important;
    height: auto !important;
}
.media-window img,
.media-window video {
    max-height: 60vh !important;
    object-fit: contain;
    width: 100%;
}
.scrollable-gallery {
    overflow-y: auto !important;
}
.scrollable-gallery .grid-wrap {
    max-height: none !important;
    overflow: visible !important;
}
.scrollable-gallery-sm {
    max-height: 300px !important;
    overflow-y: auto !important;
}
.scrollable-gallery-lg {
    max-height: 800px !important;
    overflow-y: auto !important;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Single Image"):
                    bw_image = gr.Image(label='Black and White Input Image', elem_classes="media-window")
                    
                    with gr.Row():
                        btn = gr.Button('Convert using DDColor', variant='primary', size="sm")
                        clear_btn = gr.Button('Clear', variant='secondary', scale=0, size="sm")

                with gr.TabItem("Batch"):
                    batch_files = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath"
                    )
                    batch_folder = gr.Textbox(
                        label="Or enter folder path",
                        placeholder="C:\\path\\to\\images or /path/to/images",
                        info="All valid images in folder will be processed"
                    )
                    batch_btn = gr.Button('Process Batch', variant='primary', size="sm")

                with gr.TabItem("Video (Experimental WIP)"):
                    video_input = gr.Video(label="Input Video", elem_classes="media-window")
                    gr.Markdown("⚠️ *Experimental: Model isn't trained for video, results may vary. Uses ffmpeg.*")
                    video_btn = gr.Button('Colorize Video', variant='primary', size="sm")

            input_size_radio = gr.Radio(
                choices=[512, 768, 1024],
                value=512,
                label="Processing Size",
                info="Higher = potentially better quality, more VRAM (~5GB/7GB/9GB). Changing size will reload the model - 1st gen always slow."
            )

            # Test images accordion
            with gr.Accordion("📁 Sample Test Images", open=False):
                gr.Markdown("*Click an image to load it as input*")
                test_gallery = gr.Gallery(
                    value=get_test_images(),
                    label="Test Images",
                    columns=4,
                    height=200,
                    object_fit="contain",
                    allow_preview=False,
                    elem_classes="scrollable-gallery scrollable-gallery-sm"
                )
                
        with gr.Column():
            with gr.Tabs() as output_tabs:
                with gr.TabItem("Single Result", id="single_tab"):
                    col_image_slider = ImageSlider(position=0.5,
                        label="Before/After Comparison",
                        interactive=False,
                        elem_classes="media-window"
                    )
                
                with gr.TabItem("Batch Results", id="batch_tab"):
                    batch_results_gallery = gr.Gallery(
                        label="Batch Results",
                        columns=4,
                        height=400,
                        object_fit="contain",
                        elem_classes="scrollable-gallery"
                    )
                
                with gr.TabItem("Video Result", id="video_tab"):
                    video_output = gr.Video(label="Colorized Video", elem_classes="media-window")
            
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
                allow_preview=True,
                elem_classes="scrollable-gallery scrollable-gallery-lg"
            )

    def on_clear_temp_change(value):
        """Save clear temp preference when changed"""
        settings['clear_temp_on_start'] = value
        save_settings(settings)
    
    def on_autosave_change(value):
        """Save autosave preference when changed"""
        settings['autosave'] = value
        save_settings(settings)
    
    # Event handlers - switch to single result tab first, then colorize
    btn.click(
        lambda: gr.update(selected="single_tab"),
        None,
        output_tabs
    ).then(
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
    
    # Batch processing - switch to batch results tab first
    batch_btn.click(
        lambda: gr.update(selected="batch_tab"),
        None,
        output_tabs
    ).then(
        process_batch,
        [batch_files, batch_folder, autosave_checkbox, input_size_radio],
        [batch_results_gallery]
    ).then(
        update_status_and_gallery,
        None,
        [save_status, output_gallery]
    )
    
    # Video processing - switch to video result tab first
    video_btn.click(
        lambda: gr.update(selected="video_tab"),
        None,
        output_tabs
    ).then(
        process_video,
        [video_input, input_size_radio],
        [video_output]
    ).then(
        update_status_and_gallery,
        None,
        [save_status, output_gallery]
    )

demo.launch()
