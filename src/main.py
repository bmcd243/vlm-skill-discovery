import threading
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
# import torch.optim as optim
# import torch.nn.functional as F
# import gymnasium as gym
# import gymnasium_robotics
# import numpy as np
# from collections import deque
# import cv2
# import os

# from config import Config
# from buffer import SharedReplayBuffer
# from models import Actor, Critic, Discriminator
# from vlm import InternVLWrapper
# from worker import vlm_worker_loop

# def make_env(env_id, img_size):
#     env = gym.make(env_id, render_mode=Config.RENDER_MODE)
#     return env

# def get_obs(env, img_size):
#     img = env.render() 
#     if img is None: 
#         return np.zeros((3, img_size, img_size), dtype=np.uint8)
#     img = cv2.resize(img, (img_size, img_size))
#     img = np.transpose(img, (2, 0, 1))
#     return img

# def get_frame(env, img_size):
#     """Get RGB frame for video (H, W, 3)."""
#     img = env.render()
#     if img is None:
#         return np.zeros((img_size, img_size, 3), dtype=np.uint8)
#     img = cv2.resize(img, (img_size, img_size))
#     return img

# def main():
#     Config.makedirs()
#     env = make_env(Config.ENV_ID, Config.IMG_SIZE)
    
#     # Dimensions
#     obs_shape = (Config.FRAME_STACK * 3, Config.IMG_SIZE, Config.IMG_SIZE)
#     action_dim = env.action_space.shape[0]
    
#     # Initialize Networks
#     # Note: Actor/Critic now take Skill Vector as input
#     actor = Actor(obs_shape, Config.NUM_SKILLS, action_dim).to(Config.DEVICE)
#     critic = Critic(obs_shape, Config.NUM_SKILLS, action_dim).to(Config.DEVICE)
#     discriminator = Discriminator(Config.EMBEDDING_DIM, Config.NUM_SKILLS).to(Config.DEVICE)
    
#     actor_opt = optim.Adam(actor.parameters(), lr=Config.LR)
#     critic_opt = optim.Adam(critic.parameters(), lr=Config.LR)
#     discrim_opt = optim.Adam(discriminator.parameters(), lr=Config.LR)
    
#     # Buffer stores Embeddings, not Rewards directly
#     buffer = SharedReplayBuffer(100_000, obs_shape, action_dim, Config.NUM_SKILLS, Config.EMBEDDING_DIM, Config.DEVICE)
    
#     # Initialize InternVL
#     vlm_wrapper = InternVLWrapper(Config.MODEL_PATH, Config.OUTPUT_DIR)
    
#     # Start Worker
#     stop_event = threading.Event()
#     worker_t = threading.Thread(target=vlm_worker_loop, args=(stop_event, buffer, vlm_wrapper, Config))
#     worker_t.start()
    
#     # --- Training Loop ---
#     state_deque = deque(maxlen=Config.FRAME_STACK)
    
#     # Init Env
#     _ = env.reset()
#     init_img = get_obs(env, Config.IMG_SIZE)
#     for _ in range(Config.FRAME_STACK): state_deque.append(init_img)
    
#     # Pick Initial Skill (One-Hot) and start episode tracking
#     current_skill_idx = np.random.randint(0, Config.NUM_SKILLS)
#     current_skill = np.zeros(Config.NUM_SKILLS)
#     current_skill[current_skill_idx] = 1.0
#     skill_step_count = 0  # Track how long we've executed current skill
    
#     buffer.start_episode(current_skill)
    
#     print(f"Starting DLSD (Discovery) Training with {Config.NUM_SKILLS} skills...")
#     print(f"Skill Duration: {Config.SKILL_DURATION} steps")
    
#     for step in range(Config.TOTAL_STEPS):
#         # 1. State Construction
#         current_state = np.concatenate(list(state_deque), axis=0)
        
#         # 2. Select Action (Conditioned on Skill)
#         with torch.no_grad():
#             s_t = torch.FloatTensor(current_state).unsqueeze(0).to(Config.DEVICE) / 255.0
#             z_t = torch.FloatTensor(current_skill).unsqueeze(0).to(Config.DEVICE)
#             action = actor.sample(s_t, z_t).cpu().numpy()[0]
            
#         # 3. Env Step
#         _, _, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
        
#         next_img = get_obs(env, Config.IMG_SIZE)
#         frame_rgb = get_frame(env, Config.IMG_SIZE)  # For video
#         next_deque = state_deque.copy()
#         next_deque.append(next_img)
#         next_state = np.concatenate(list(next_deque), axis=0)
        
#         # 4. Add to Buffer with frame
#         buffer.add(current_state, action, next_state, current_skill, done, frame=frame_rgb)
        
#         state_deque = next_deque
#         skill_step_count += 1
        
#         # Check if we should change skills (after SKILL_DURATION steps or episode end)
#         should_resample_skill = (skill_step_count >= Config.SKILL_DURATION) or done
        
#         # 5. Updates (Only using VLM-Embedded data)
#         if step > Config.WARMUP_STEPS:
#             batch = buffer.sample_labeled(Config.BATCH_SIZE)
            
#             if batch is not None:
#                 b_s, b_a, b_ns, b_d, b_z, b_emb = batch
                
#                 # --- A. Calculate Intrinsic Reward ---
#                 # R = log D(z|emb) - log(1/K)
#                 logits = discriminator(b_emb)
#                 log_probs = F.log_softmax(logits, dim=1)
                
#                 # Get log_prob of the actual skill executed
#                 # b_z is one-hot, so we can multiply or use indices
#                 skill_indices = torch.argmax(b_z, dim=1).unsqueeze(1)
#                 selected_log_probs = log_probs.gather(1, skill_indices)
                
#                 prior_log_prob = np.log(1.0 / Config.NUM_SKILLS)
#                 intrinsic_reward = selected_log_probs - prior_log_prob
                
#                 # Total Reward (Pure Discovery)
#                 total_reward = intrinsic_reward
                
#                 # --- B. Discriminator Update ---
#                 # Minimize CrossEntropy(logits, skill_indices)
#                 discrim_loss = F.cross_entropy(logits, skill_indices.squeeze(1))
                
#                 discrim_opt.zero_grad()
#                 discrim_loss.backward()
#                 discrim_opt.step()
                
#                 # --- C. Critic Update ---
#                 with torch.no_grad():
#                     next_action = actor.sample(b_ns, b_z)
#                     target_q1, target_q2 = critic(b_ns, b_z, next_action)
#                     target_q = torch.min(target_q1, target_q2)
#                     target_val = total_reward + (1 - b_d) * Config.GAMMA * target_q
                    
#                 q1, q2 = critic(b_s, b_z, b_a)
#                 q_loss = F.mse_loss(q1, target_val) + F.mse_loss(q2, target_val)
                
#                 critic_opt.zero_grad()
#                 q_loss.backward()
#                 critic_opt.step()
                
#                 # --- D. Actor Update ---
#                 new_action = actor.sample(b_s, b_z)
#                 q1_new, q2_new = critic(b_s, b_z, new_action)
#                 q_new = torch.min(q1_new, q2_new)
#                 actor_loss = -q_new.mean()
                
#                 actor_opt.zero_grad()
#                 actor_loss.backward()
#                 actor_opt.step()
                
#         # Handle skill resampling and episode boundaries
#         if should_resample_skill:
#             # Finish current episode
#             buffer.finish_episode()
            
#             # Resample skill
#             current_skill_idx = np.random.randint(0, Config.NUM_SKILLS)
#             current_skill = np.zeros(Config.NUM_SKILLS)
#             current_skill[current_skill_idx] = 1.0
#             skill_step_count = 0
            
#             # Start new episode
#             buffer.start_episode(current_skill)
        
#         if done:
#             _ = env.reset()
#             state_deque.clear()
#             init_img = get_obs(env, Config.IMG_SIZE)
#             for _ in range(Config.FRAME_STACK): state_deque.append(init_img)
            
#         if step % 100 == 0:
#             labeled_cnt = buffer.get_labeled_episode_count()
#             total_episodes = len(buffer.episodes)
#             print(f"Step {step} | Episodes: {total_episodes} | Labeled: {labeled_cnt}")

#     print("Training Finished.")
#     stop_event.set()
#     worker_t.join()

# if __name__ == "__main__":
#     main()
