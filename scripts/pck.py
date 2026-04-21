import numpy as np
from model import Model
import torch
import argparse
from dataset import create_dataloader
from pathlib import Path

# Returns number of keypoints within threshold of true position
def keypoints_within_threshold(pred, true, threshold=10):
    # pred and true are (B, 17, 2)
    distances = torch.norm(pred - true, dim=2)  # (B, 17)
    within_threshold = distances < threshold
    correct = within_threshold.sum().item()
    return correct

def percentage_correct_keypoints(model, dataloader, threshold=10):
    model.eval()
    total_correct = 0
    total_keypoints = 0

    mae = 0.0

    with torch.no_grad():
        for radar, true_pose in dataloader:
            pred_pose = model(radar)
            total_correct += keypoints_within_threshold(pred_pose, true_pose, threshold)
            mae += torch.mean(torch.abs(pred_pose - true_pose)).item() * pred_pose.shape[0] * pred_pose.shape[1]
            total_keypoints += pred_pose.shape[0] * pred_pose.shape[1]

    mae /= total_keypoints
    precision = total_correct / (total_correct + (total_keypoints - total_correct)) if (total_correct + (total_keypoints - total_correct)) > 0 else 0
    recall = total_correct / total_keypoints if total_keypoints > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f'MAE: {mae:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

    return total_correct / total_keypoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--data_path', type=str, default='P1/')
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    dataloader = create_dataloader(root_path=args.data_path, num_samples=args.num_samples, batch_size=args.batch_size, shuffle=False)
    pck = percentage_correct_keypoints(model, dataloader, threshold=args.threshold)
    print(f'PCK @ {args.threshold} pixels: {pck:.4f}')
