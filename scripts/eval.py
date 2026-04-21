import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from model import Model
import torch
import argparse
from dataset import create_dataloader
from pathlib import Path
from pck import keypoints_within_threshold

# joints connections for 2d keypoints
connections = np.array([[13, 15], [11, 13], [14, 16], [12, 14], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]])

def load_mask(path):
    path = Path(path)
    mask_path = path.with_name(path.name.replace('_radar.npz', '_mask.npz'))
    mask_data = np.load(mask_path)
    mask = torch.from_numpy(mask_data['mask']).float()
    return mask

# Function to allow us to visually see pose predictions
def visual_eval(model, data, path, threshold=10.0, save_graph=False, graph_path=None):
    radar, pose = data
    mask = load_mask(path)
    output = model(radar)

    # Create a figure with two subplots        
    pose_x = pose[0, :, 0].detach().numpy()
    pose_y = pose[0, :, 1].detach().numpy()
    pred_pose_x = output[0, :, 0].detach().numpy()
    pred_pose_y = output[0, :, 1].detach().numpy()

    pck = keypoints_within_threshold(output, pose, threshold=args.threshold) / (output.shape[0] * output.shape[1])
    print(f'PCK @ {args.threshold} pixels: {pck:.4f}')

    plt.imshow(mask[0, :, :], cmap='gray')

    # Plot pose on the right subplot
    for connection in connections:
        x = pred_pose_x[connection]
        y = pred_pose_y[connection]
        plt.plot(x, y,color='r',marker='.')
        x = pose_x[connection]
        y = pose_y[connection]
        plt.plot(x, y,color='g',marker='.')
    
    # Add PCK text
    plt.text(10, 25, f'PCK @ {threshold} pixels: {pck:.4f}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    plt.legend(['pred','true'])
    plt.title('Pose')

    if args.save_graph:
        plt.savefig(graph_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--data_path', type=str, default='test_example/')
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--save_graph', action='store_true', default=False)
    parser.add_argument('--graph_path', type=str, default=None)
    args = parser.parse_args()
    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    dataloader, dataset = create_dataloader(root_path=args.data_path, num_samples=args.num_samples, batch_size=1, shuffle=False)
    for data, i in zip(dataloader, range(len(dataset))):
        path = dataset.radar_files[i]
        if args.graph_path is not None:
            graph_path = Path(args.graph_path).with_name(f'{Path(args.graph_path).stem}_{i}.png')
        else:
            graph_path = path.with_name(path.name.replace('_radar.npz', '_eval.png'))
        visual_eval(model, data, path, threshold=args.threshold, save_graph=args.save_graph, graph_path=graph_path)
