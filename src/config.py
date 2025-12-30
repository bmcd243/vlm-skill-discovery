import torch
import os

class Config:
    # Environment
    ENV_ID = "FetchPush-v4" 
    RENDER_MODE = "rgb_array"
    FRAME_STACK = 3           
    IMG_SIZE = 224            
    ACTION_REPEAT = 1
    
    # Skill Discovery (DIAYN / DLSD)
    NUM_SKILLS = 2             # Number of discrete skills to discover
    EMBEDDING_DIM = 384        # Dimension of S-BERT embeddings
    SKILL_DURATION = 50        # How many steps to execute each skill before resampling
    
    # Paths & Models
    OUTPUT_DIR = "./output_videos"
    MODEL_PATH = "OpenGVLab/InternVL3-8B"
    
    # Training
    TOTAL_STEPS = 50_000
    WARMUP_STEPS = 1_000
    BATCH_SIZE = 64
    LR = 3e-4
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.1                # Entropy regularization
    
    # Asynchronous VLM Worker
    VLM_BATCH_SIZE = 2         # Low batch size to save VRAM
    VLM_POLL_INTERVAL = 1.0    
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def makedirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
