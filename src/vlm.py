import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import imageio.v3 as iio
import numpy as np
import gc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVLWrapper:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        
        print("Loading SentenceTransformer (Embedding Model)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Loading InternVL Model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

    def build_transform(self, input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_video_from_file(self, video_path, input_size=448, max_num=1, num_segments=8):
        # Read all frames from video using imageio
        frames = iio.imread(video_path, plugin='pyav')
        max_frame = len(frames) - 1
        
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = np.linspace(0, max_frame, num_segments, dtype=int)
        
        for idx in frame_indices:
            img = Image.fromarray(frames[idx]).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def get_embedding(self, video_path):
        # 1. Generate Description
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            pixel_values, num_patches_list = self.load_video_from_file(
                video_path, num_segments=8, max_num=1
            )
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            
            # --- PROMPT FOR DISCOVERY ---
            # Task-agnostic prompt for discovering diverse behaviors
            question = (
                video_prefix + 
                "Describe the distinct motion pattern or behavior the robot performs in this video. "
                "Focus on the type of movement, direction, speed, and interaction style."
            )
            
            generation_config = dict(max_new_tokens=128, do_sample=False)
            
            description, _ = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                question, 
                generation_config,
                num_patches_list=num_patches_list, 
                history=None, 
                return_history=True
            )
            
            # 2. Encode Description to Vector
            embedding = self.embedder.encode(description)
            return embedding, description
            
        except Exception as e:
            print(f"[VLM Error] {e}")
            return np.zeros(384), "Error"
