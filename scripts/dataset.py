import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class RadarDataset(Dataset):
    def __init__(self, root_path):
        """
        Searches recursively for all radar frames in the data folder.
        Expects radar files to end in `_radar.npz`.
        """
        self.root_path = Path(root_path)
        self.radar_files = sorted(list(self.root_path.rglob('*_radar.npz')))
        
    def __len__(self):
        return len(self.radar_files)
        
    def __getitem__(self, idx):
        radar_path = self.radar_files[idx]
        pose_path = radar_path.with_name(radar_path.name.replace('_radar.npz', '_pose.npz'))
        
        radar_data = np.load(radar_path)
        hori = radar_data['hm_hori']
        vert = radar_data['hm_vert']
        
        # Replace NaNs with 0.0 and convert to tensors with channel dim
        hori = torch.from_numpy(hori).float().nan_to_num(0.0).unsqueeze(0) # [1, 256, 128]
        vert = torch.from_numpy(vert).float().nan_to_num(0.0).unsqueeze(0) # [1, 256, 128]
        
        # Concatenate into shape [2, 256, 128]
        radar_tensor = torch.cat((hori, vert), dim=0)
        
        if pose_path.exists():
            pose_data = np.load(pose_path)
            kp = pose_data['kp']
            kp_tensor = torch.from_numpy(kp).float()
            return radar_tensor, kp_tensor
            
        return radar_tensor

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

def create_dataloader(root_path, batch_size=32, shuffle=True) -> DataLoader:
    dataset = RadarDataset(root_path)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=radar_collate_fn
    )