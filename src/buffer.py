import numpy as np
import torch
import threading

class SharedReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, skill_dim, embedding_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()

        # RL Data (Transition-level for training)
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.skills = np.zeros((capacity, skill_dim), dtype=np.float32)  # One-hot
        
        # Episode-level tracking
        self.episodes = []  # List of {"start": idx, "end": idx, "skill": z, "frames": [], "embedding": None, "is_embedded": False}
        self.current_episode = None
        self.unlabeled_episodes = []  # Indices into self.episodes

    def start_episode(self, skill):
        """Start tracking a new episode."""
        with self.lock:
            self.current_episode = {
                "start": self.ptr,
                "skill": skill,
                "frames": [],
                "embedding": None,
                "is_embedded": False
            }
    
    def add(self, state, action, next_state, skill, done, frame=None):
        """Add transition and optionally track frame for episode."""
        with self.lock:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.next_states[self.ptr] = next_state
            self.skills[self.ptr] = skill
            self.dones[self.ptr] = done
            
            # Track frame for current episode
            if self.current_episode is not None and frame is not None:
                self.current_episode["frames"].append(frame)
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def finish_episode(self):
        """Mark episode as complete and queue for VLM processing."""
        with self.lock:
            if self.current_episode is not None and len(self.current_episode["frames"]) > 0:
                self.current_episode["end"] = self.ptr - 1
                episode_idx = len(self.episodes)
                self.episodes.append(self.current_episode)
                self.unlabeled_episodes.append(episode_idx)
                self.current_episode = None

    def sample_labeled(self, batch_size):
        """Sample transitions from episodes that have VLM embeddings."""
        with self.lock:
            # Get all transitions from labeled episodes
            labeled_indices = []
            for ep in self.episodes:
                if ep["is_embedded"]:
                    # Add all transitions from this episode
                    for i in range(ep["start"], ep["end"] + 1):
                        if i < self.size:
                            labeled_indices.append(i)
            
            if len(labeled_indices) < batch_size:
                return None
            
            idxs = np.random.choice(labeled_indices, batch_size, replace=len(labeled_indices) < batch_size)
            
            # Get embeddings from episodes
            embeddings = []
            for idx in idxs:
                # Find which episode this transition belongs to
                for ep in self.episodes:
                    if ep["is_embedded"] and ep["start"] <= idx <= ep["end"]:
                        embeddings.append(ep["embedding"])
                        break
            
            return (
                torch.FloatTensor(self.states[idxs]).to(self.device) / 255.0,
                torch.FloatTensor(self.actions[idxs]).to(self.device),
                torch.FloatTensor(self.next_states[idxs]).to(self.device) / 255.0,
                torch.FloatTensor(self.dones[idxs]).to(self.device),
                torch.FloatTensor(self.skills[idxs]).to(self.device),
                torch.FloatTensor(np.array(embeddings)).to(self.device)
            )

    def get_unlabeled_episodes(self, batch_size):
        """Get unlabeled episodes for VLM processing."""
        with self.lock:
            if len(self.unlabeled_episodes) < batch_size:
                return None
            
            # Get episode indices
            ep_indices = self.unlabeled_episodes[:batch_size]
            episodes_data = []
            
            for idx in ep_indices:
                ep = self.episodes[idx]
                episodes_data.append({
                    "idx": idx,
                    "frames": ep["frames"],
                    "skill": ep["skill"]
                })
            
            return episodes_data
    
    def update_episode_embedding(self, episode_idx, embedding):
        """Update episode with VLM embedding."""
        with self.lock:
            if episode_idx < len(self.episodes):
                self.episodes[episode_idx]["embedding"] = embedding
                self.episodes[episode_idx]["is_embedded"] = True
                if episode_idx in self.unlabeled_episodes:
                    self.unlabeled_episodes.remove(episode_idx)
    
    def get_labeled_episode_count(self):
        """Count episodes with embeddings."""
        with self.lock:
            return sum(1 for ep in self.episodes if ep["is_embedded"])
