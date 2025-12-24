import os
import json
import copy
import math
import time
import random
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch
from PIL import Image
import gradio as gr
import spaces
from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler)
from huggingface_hub import (
    hf_hub_download,
    HfFileSystem,
    ModelCard,
    snapshot_download)
from diffusers.utils import load_image
import requests
from urllib.parse import urlparse
from typing import Iterable
import tempfile
import shutil
import uuid
import zipfile

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red, # Use the new color
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# META: CUDA_CHECK / GPU_INFO
device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

loras = [
    # Sample Qwen-compatible LoRAs
    {
        "image": "https://huggingface.co/prithivMLmods/Qwen-Image-Studio-Realism/resolve/main/images/2.png",
        "title": "Studio Realism",
        "repo": "prithivMLmods/Qwen-Image-Studio-Realism",
        "weights": "qwen-studio-realism.safetensors",
        "trigger_word": "Studio Realism"
    },
    {
        "image": "https://huggingface.co/prithivMLmods/Qwen-Image-Sketch-Smudge/resolve/main/images/1.png",
        "title": "Sketch Smudge",
        "repo": "prithivMLmods/Qwen-Image-Sketch-Smudge",
        "weights": "qwen-sketch-smudge.safetensors",
        "trigger_word": "Sketch Smudge"
    },
    {
        "image": "https://huggingface.co/Shakker-Labs/AWPortrait-QW/resolve/main/images/08fdaf6b644b61136340d5c908ca37993e47f34cdbe2e8e8251c4c72.jpg",
        "title": "AWPortrait QW",
        "repo": "Shakker-Labs/AWPortrait-QW",
        "weights": "AWPortrait-QW_1.0.safetensors",
        "trigger_word": "Portrait"
    },
    {
        "image": "https://huggingface.co/prithivMLmods/Qwen-Image-Anime-LoRA/resolve/main/images/1.png",
        "title": "Qwen Anime",
        "repo": "prithivMLmods/Qwen-Image-Anime-LoRA",
        "weights": "qwen-anime.safetensors",
        "trigger_word": "Qwen Anime"
    },
    {
        "image": "https://huggingface.co/flymy-ai/qwen-image-realism-lora/resolve/main/assets/flymy_realism.png",
        "title": "Image Realism",
        "repo": "flymy-ai/qwen-image-realism-lora",
        "weights": "flymy_realism.safetensors",
        "trigger_word": "Super Realism Portrait"
    },
    {
        "image": "https://huggingface.co/prithivMLmods/Qwen-Image-Fragmented-Portraiture/resolve/main/images/3.png",
        "title": "Fragmented Portraiture",
        "repo": "prithivMLmods/Qwen-Image-Fragmented-Portraiture",
        "weights": "qwen-fragmented-portraiture.safetensors",
        "trigger_word": "Fragmented Portraiture"
    },
    {
        "image": "https://huggingface.co/prithivMLmods/Qwen-Image-Synthetic-Face/resolve/main/images/2.png",
        "title": "Synthetic Face",
        "repo": "prithivMLmods/Qwen-Image-Synthetic-Face",
        "weights": "qwen-synthetic-face.safetensors",
        "trigger_word": "Synthetic Face"
    },
    {
        "image": "https://huggingface.co/itspoidaman/qwenglitch/resolve/main/images/GyZTwJIbkAAhS4h.jpeg",
        "title": "Qwen Glitch",
        "repo": "itspoidaman/qwenglitch",
        "weights": "qwenglitch1.safetensors",
        "trigger_word": "qwenglitch"
    },
    {
        "image": "https://huggingface.co/alfredplpl/qwen-image-modern-anime-lora/resolve/main/sample1.jpg",
        "title": "Modern Anime Lora",
        "repo": "alfredplpl/qwen-image-modern-anime-lora",
        "weights": "lora.safetensors",
        "trigger_word": "Japanese modern anime style"
    },
]

# Initialize the base model
dtype = torch.bfloat16
base_model = "Qwen/Qwen-Image"

# Scheduler configuration from the Qwen-Image-Lightning repository
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    base_model, scheduler=scheduler, torch_dtype=dtype
).to(device)

# Lightning LoRA info (no global state)
LIGHTNING_LORA_REPO = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_LORA_WEIGHT = "Qwen-Image-Lightning-8steps-V1.0.safetensors"

MAX_SEED = np.iinfo(np.int32).max

class Timer:
    def __init__(self, task_name=""):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.task_name:
            print(f"Elapsed time for {self.task_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")

def compute_image_dimensions(aspect_ratio):
    """Converts aspect ratio string to width, height tuple."""
    if aspect_ratio == "1:1":
        return 1024, 1024
    elif aspect_ratio == "16:9":
        return 1152, 640
    elif aspect_ratio == "9:16":
        return 640, 1152
    elif aspect_ratio == "4:3":
        return 1024, 768
    elif aspect_ratio == "3:4":
        return 768, 1024
    elif aspect_ratio == "3:2":
        return 1024, 688
    elif aspect_ratio == "2:3":
        return 688, 1024
    else:
        return 1024, 1024

def handle_lora_selection(evt: gr.SelectData, aspect_ratio):
    selected_lora = loras[evt.index]
    new_placeholder = f"Type a prompt for {selected_lora['title']}"
    lora_repo = selected_lora["repo"]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo}) ✅"
    
    # Update aspect ratio if specified in LoRA config
    if "aspect" in selected_lora:
        if selected_lora["aspect"] == "portrait":
            aspect_ratio = "9:16"
        elif selected_lora["aspect"] == "landscape":
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"
    
    return (
        gr.update(placeholder=new_placeholder),
        updated_text,
        evt.index,
        aspect_ratio,
    )

def adjust_generation_mode(speed_mode):
    """Update UI based on speed/quality toggle."""
    if speed_mode == "Fast (8 steps)":
        return gr.update(value="Fast mode selected - 8 steps with Lightning LoRA"), 8, 1.0
    else: 
        return gr.update(value="Base mode selected - 50 steps for best quality"), 50, 4.0

@spaces.GPU(duration=90)
def create_image(prompt_mash, steps, seed, cfg_scale, width, height, lora_scale, negative_prompt=""):
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    with Timer("Generating image"):
        # Generate image
        image = pipe(
            prompt=prompt_mash,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            true_cfg_scale=cfg_scale,  # Use true_cfg_scale for Qwen-Image
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        
    return image

@spaces.GPU(duration=90)
def process_adapter_generation(prompt, cfg_scale, steps, selected_index, randomize_seed, seed, aspect_ratio, lora_scale, speed_mode, progress=gr.Progress(track_tqdm=True)):
    if selected_index is None:
        raise gr.Error("You must select a LoRA before proceeding.")
    
    selected_lora = loras[selected_index]
    lora_path = selected_lora["repo"]
    trigger_word = selected_lora["trigger_word"]
    
    # Prepare prompt with trigger word
    if trigger_word:
        if "trigger_position" in selected_lora:
            if selected_lora["trigger_position"] == "prepend":
                prompt_mash = f"{trigger_word} {prompt}"
            else:
                prompt_mash = f"{prompt} {trigger_word}"
        else:
            prompt_mash = f"{trigger_word} {prompt}"
    else:
        prompt_mash = prompt

    # Always unload any existing LoRAs first to avoid conflicts
    with Timer("Unloading existing LoRAs"):
        pipe.unload_lora_weights()

    # Load LoRAs based on speed mode
    if speed_mode == "Fast (8 steps)":
        with Timer("Loading Lightning LoRA and style LoRA"):
            # Load Lightning LoRA first
            pipe.load_lora_weights(
                LIGHTNING_LORA_REPO, 
                weight_name=LIGHTNING_LORA_WEIGHT,
                adapter_name="lightning"
            )
            
            # Load the selected style LoRA
            weight_name = selected_lora.get("weights", None)
            pipe.load_lora_weights(
                lora_path, 
                weight_name=weight_name, 
                low_cpu_mem_usage=True,
                adapter_name="style"
            )
            
            # Set both adapters active with their weights
            pipe.set_adapters(["lightning", "style"], adapter_weights=[1.0, lora_scale])
    else:
        # Quality mode - only load the style LoRA
        with Timer(f"Loading LoRA weights for {selected_lora['title']}"):
            weight_name = selected_lora.get("weights", None)
            pipe.load_lora_weights(
                lora_path, 
                weight_name=weight_name, 
                low_cpu_mem_usage=True,
                adapter_name="style"
            )
            pipe.set_adapters(["style"], adapter_weights=[lora_scale])
                
    # Set random seed for reproducibility
    with Timer("Randomizing seed"):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
    
    # Get image dimensions from aspect ratio
    width, height = compute_image_dimensions(aspect_ratio)
    
    # Generate the image
    final_image = create_image(prompt_mash, steps, seed, cfg_scale, width, height, lora_scale)
    
    return final_image, seed

def fetch_hf_adapter_files(link):
    split_link = link.split("/")
    if len(split_link) != 2:
        raise Exception("Invalid Hugging Face repository link format.")

    print(f"Repository attempted: {split_link}")
    
    # Load model card
    model_card = ModelCard.load(link)
    base_model = model_card.data.get("base_model")
    print(f"Base model: {base_model}")

    # Validate model type (for Qwen-Image)
    acceptable_models = {"Qwen/Qwen-Image"}
    
    models_to_check = base_model if isinstance(base_model, list) else [base_model]
    
    if not any(model in acceptable_models for model in models_to_check):
        raise Exception("Not a Qwen-Image LoRA!")
        
    # Extract image and trigger word
    image_path = model_card.data.get("widget", [{}])[0].get("output", {}).get("url", None)
    trigger_word = model_card.data.get("instance_prompt", "")
    image_url = f"https://huggingface.co/{link}/resolve/main/{image_path}" if image_path else None

    # Initialize Hugging Face file system
    fs = HfFileSystem()
    try:
        list_of_files = fs.ls(link, detail=False)
        
        # Find safetensors file
        safetensors_name = None
        for file in list_of_files:
            filename = file.split("/")[-1]
            if filename.endswith(".safetensors"):
                safetensors_name = filename
                break

        if not safetensors_name:
            raise Exception("No valid *.safetensors file found in the repository.")

    except Exception as e:
        print(e)
        raise Exception("You didn't include a valid Hugging Face repository with a *.safetensors LoRA")
    
    return split_link[1], link, safetensors_name, trigger_word, image_url

def validate_custom_adapter(link):
    print(f"Checking a custom model on: {link}")
    
    if link.endswith('.safetensors'):
        if 'huggingface.co' in link:
            parts = link.split('/')
            try:
                hf_index = parts.index('huggingface.co')
                username = parts[hf_index + 1]
                repo_name = parts[hf_index + 2]
                repo = f"{username}/{repo_name}"
                
                safetensors_name = parts[-1]
                
                try:
                    model_card = ModelCard.load(repo)
                    trigger_word = model_card.data.get("instance_prompt", "")
                    image_path = model_card.data.get("widget", [{}])[0].get("output", {}).get("url", None)
                    image_url = f"https://huggingface.co/{repo}/resolve/main/{image_path}" if image_path else None
                except:
                    trigger_word = ""
                    image_url = None
                
                return repo_name, repo, safetensors_name, trigger_word, image_url
            except:
                raise Exception("Invalid safetensors URL format")
    
    if link.startswith("https://"):
        if link.startswith("https://huggingface.co") or link.startswith("https://www.huggingface.co"):
            link_split = link.split("huggingface.co/")
            return fetch_hf_adapter_files(link_split[1])
    else: 
        return fetch_hf_adapter_files(link)

def incorporate_custom_adapter(custom_lora):
    global loras
    if custom_lora:
        try:
            title, repo, path, trigger_word, image = validate_custom_adapter(custom_lora)
            print(f"Loaded custom LoRA: {repo}")
            card = f'''
            <div class="custom_lora_card">
              <span>Loaded custom LoRA:</span>
              <div class="card_internal">
                <img src="{image}" />
                <div>
                    <h3>{title}</h3>
                    <small>{"Using: <code><b>"+trigger_word+"</code></b> as the trigger word" if trigger_word else "No trigger word found. If there's a trigger word, include it in your prompt"}<br></small>
                </div>
              </div>
            </div>
            '''
            existing_item_index = next((index for (index, item) in enumerate(loras) if item['repo'] == repo), None)
            if existing_item_index is None:
                new_item = {
                    "image": image,
                    "title": title,
                    "repo": repo,
                    "weights": path,
                    "trigger_word": trigger_word
                }
                print(new_item)
                loras.append(new_item)
                existing_item_index = len(loras) - 1  # Get the actual index after adding
        
            return gr.update(visible=True, value=card), gr.update(visible=True), gr.Gallery(selected_index=None), f"Custom: {path}", existing_item_index, trigger_word
        except Exception as e:
            gr.Warning(f"Invalid LoRA: either you entered an invalid link, or a non-Qwen-Image LoRA, this was the issue: {e}")
            return gr.update(visible=True, value=f"Invalid LoRA: either you entered an invalid link, a non-Qwen-Image LoRA"), gr.update(visible=True), gr.update(), "", None, ""
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

def discard_custom_adapter():
    return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

process_adapter_generation.zerogpu = True

css = '''
#gen_btn{height: 100%}
#gen_column{align-self: stretch}
#title{text-align: center}
#title h1{font-size: 3em; display:inline-flex; align-items:center}
#title img{width: 100px; margin-right: 0.5em}
#gallery .grid-wrap{height: 10vh}
#lora_list{background: var(--block-background-fill);padding: 0 1em .3em; font-size: 90%}
.card_internal{display: flex;height: 100px;margin-top: .5em}
.card_internal img{margin-right: 1em}
.styler{--form-gap-width: 0px !important}
#speed_status{padding: .5em; border-radius: 5px; margin: 1em 0}
'''

with gr.Blocks(delete_cache=(240, 240)) as app:
    title = gr.HTML("""<h1>Qwen Image LoRA DLC⛵</h1>""", elem_id="title")
    selected_index = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=1, placeholder="✦︎ Choose the LoRA and type the prompt")
        with gr.Column(scale=1, elem_id="gen_column"):
            generate_button = gr.Button("Generate", variant="primary", elem_id="gen_btn")
    
    with gr.Row():
        with gr.Column():
            selected_info = gr.Markdown("")
            gallery = gr.Gallery(
                [(item["image"], item["title"]) for item in loras],
                label="LoRA Gallery",
                allow_preview=False,
                columns=3,
                elem_id="gallery",
                #show_share_button=False
            )
            with gr.Group():
                custom_lora = gr.Textbox(label="Custom LoRA", placeholder="username/lora-model-name")
                gr.Markdown("[Check Qwen-Image LoRAs](https://huggingface.co/models?other=base_model:adapter:Qwen/Qwen-Image)", elem_id="lora_list")
            custom_lora_info = gr.HTML(visible=False)
            custom_lora_button = gr.Button("Remove custom LoRA", visible=False)
        
        with gr.Column():
            result = gr.Image(label="Generated Image", format="png")

            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    value="3:2"
                    )
            with gr.Row():
                speed_mode = gr.Dropdown(
                    label="Output Mode",
                    choices=["Fast (8 steps)", "Base (50 steps)"],
                    value="Base (50 steps)",
                )
            
            speed_status = gr.Markdown("Base mode selected - 50 steps for best quality", elem_id="speed_status")

    with gr.Row():
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Column():
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="Guidance Scale (True CFG)", 
                        minimum=1.0, 
                        maximum=5.0, 
                        step=0.1, 
                        value=4.0,
                        info="Lower for speed mode, higher for quality"
                    )
                    steps = gr.Slider(
                        label="Steps", 
                        minimum=4, 
                        maximum=50, 
                        step=1, 
                        value=50,
                        info="Automatically set by speed mode"
                    )
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)
                    lora_scale = gr.Slider(label="LoRA Scale", minimum=0, maximum=2, step=0.01, value=1.0)

    # Event handlers
    gallery.select(
        handle_lora_selection,
        inputs=[aspect_ratio],
        outputs=[prompt, selected_info, selected_index, aspect_ratio]
    )
    
    speed_mode.change(
        adjust_generation_mode,
        inputs=[speed_mode],
        outputs=[speed_status, steps, cfg_scale]
    )
    
    custom_lora.input(
        incorporate_custom_adapter,
        inputs=[custom_lora],
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, prompt]
    )
    
    custom_lora_button.click(
        discard_custom_adapter,
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, custom_lora]
    )
    
    gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn=process_adapter_generation,
        inputs=[prompt, cfg_scale, steps, selected_index, randomize_seed, seed, aspect_ratio, lora_scale, speed_mode],
        outputs=[result, seed]
    )

app.queue()
app.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)

