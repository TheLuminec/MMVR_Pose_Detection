import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from model import Model
import torch
import argparse
from dataset import create_dataloader
from pathlib import Path
# joints connections for 2d keypoints
connections = np.array([[13, 15], [11, 13], [14, 16], [12, 14], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]])

def load_data_instance(path):
    path = Path(path)
    mask = np.load(list(path.rglob('*_mask.npz'))[0])
    return mask

# Function to allow us to visually see pose predictions
def visual_eval(model, path):
    radar, pose = next(iter(create_dataloader(path, batch_size=1, num_samples=1)))
    mask = load_data_instance(path)['mask']
    output = model(radar)

    # Create a figure with two subplots        
    pose_x = pose[0, :, 0].detach().numpy()
    pose_y = pose[0, :, 1].detach().numpy()
    pred_pose_x = output[0, :, 0].detach().numpy()
    pred_pose_y = output[0, :, 1].detach().numpy()

    plt.imshow(mask[0, :, :], cmap='gray')

    # Plot pose on the right subplot
    for connection in connections:
        x = pred_pose_x[connection]
        y = pred_pose_y[connection]
        plt.plot(x, y,color='r',marker='.')
        x = pose_x[connection]
        y = pose_y[connection]
        plt.plot(x, y,color='g',marker='.')
    
    plt.legend(['pred','true'])
    plt.title('Pose')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--data_path', type=str, default='test_example/')
    args = parser.parse_args()
    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    visual_eval(model, args.data_path)
