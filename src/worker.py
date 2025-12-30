import time
import os
import numpy as np
import imageio

def vlm_worker_loop(stop_event, buffer, vlm_wrapper, config):
    print(" [Worker] Started VLM Embedding Service...")
    
    temp_dir = os.path.join(config.OUTPUT_DIR, "temp_clips")
    os.makedirs(temp_dir, exist_ok=True)
    
    while not stop_event.is_set():
        # 1. Fetch episode data
        episodes_data = buffer.get_unlabeled_episodes(config.VLM_BATCH_SIZE)
        if episodes_data is None:
            time.sleep(config.VLM_POLL_INTERVAL)
            continue
        
        # 2. Process each episode
        for ep_data in episodes_data:
            episode_idx = ep_data["idx"]
            frames = ep_data["frames"]
            
            if len(frames) == 0:
                continue
            
            # Convert frames to video
            temp_path = os.path.join(temp_dir, f"episode_{episode_idx}.mp4")
            try:
                imageio.mimwrite(temp_path, frames, fps=10, format='FFMPEG', macro_block_size=1)
                
                # 3. Get Embedding from full episode
                emb, desc = vlm_wrapper.get_embedding(temp_path)
                
                # 4. Update buffer with embedding
                buffer.update_episode_embedding(episode_idx, emb)
                
                print(f" [Worker] Episode {episode_idx}: {desc[:80]}...")
                
            except Exception as e:
                print(f" [Worker] Error processing episode {episode_idx}: {e}")
            finally:
                # Clean up
                try: 
                    os.remove(temp_path)
                except: 
                    pass
