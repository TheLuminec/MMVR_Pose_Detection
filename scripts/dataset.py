import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random

class RadarDataset(Dataset):
    def __init__(self, root_path, num_samples=None):
        """
        Searches recursively for all radar frames in the data folder.
        Expects radar files to end in `_radar.npz`.
        Preloads all data into memory during initialization.
        """
        self.root_path = Path(root_path)
        self.radar_files = sorted(list(self.root_path.rglob('*_radar.npz')))
        
        if num_samples is not None and num_samples < len(self.radar_files):
            self.radar_files = random.sample(self.radar_files, num_samples)
        
        self.preloaded_radars = []
        self.preloaded_poses = []
        self.has_poses = False
        
        print(f"Preloading {len(self.radar_files)} samples into memory. Note: This may consume a significant amount of RAM.")
        for radar_path in self.radar_files:
            pose_path = radar_path.with_name(radar_path.name.replace('_radar.npz', '_pose.npz'))
            
            # Load and preprocess radar
            radar_data = np.load(radar_path)
            hori = radar_data['hm_hori']
            vert = radar_data['hm_vert']
            
            hori = torch.from_numpy(hori).float().nan_to_num(0.0).unsqueeze(0) # [1, 256, 128]
            vert = torch.from_numpy(vert).float().nan_to_num(0.0).unsqueeze(0) # [1, 256, 128]
            radar_tensor = torch.cat((hori, vert), dim=0)
            self.preloaded_radars.append(radar_tensor)
            
            # Load pose if it exists
            if pose_path.exists():
                self.has_poses = True
                pose_data = np.load(pose_path)
                kp = pose_data['kp']
                kp_tensor = torch.from_numpy(kp).float()
                self.preloaded_poses.append(kp_tensor)
        
    def __len__(self):
        return len(self.preloaded_radars)
        
    def __getitem__(self, idx):
        if self.has_poses:
            return self.preloaded_radars[idx], self.preloaded_poses[idx]
        return self.preloaded_radars[idx]

def radar_collate_fn(batch):
    """
    Custom collate_fn to handle variable 'n' objects in the pose keypoints.
    """
    radars = []
    poses = []
    
    for item in batch:
        if isinstance(item, tuple):
            radars.append(item[0])
            poses.append(item[1])
        else:
            radars.append(item)
            
    radars = torch.stack(radars, dim=0)
    
    if poses:
        # Since pose objects vary per frame, we return them as a list of tensors
        return radars, poses
        
    return radars

def create_dataloader(root_path, batch_size=32, shuffle=True, num_samples=None) -> DataLoader:
    print("Creating dataloader...")
    dataset = RadarDataset(root_path, num_samples=num_samples)
    print("Dataset created with", len(dataset), "samples.")
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=radar_collate_fn
    )