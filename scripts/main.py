from model import Model
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

if __name__ == "__main__":
    data = np.load('test_example/00383_radar.npz')
    hori = data['hm_hori']  # (256, 128)
    vert = data['hm_vert']  # (256, 128)

    data = np.load('test_example/00383_pose.npz')
    kp = data['kp']
    print("raw NaNs:", np.isnan(hori).any(), np.isnan(vert).any())

    # Conv2d expects tensors shaped as [batch, channel, height, width].
    hori = torch.from_numpy(hori).float().nan_to_num(0.0).unsqueeze(0).unsqueeze(0)
    vert = torch.from_numpy(vert).float().nan_to_num(0.0).unsqueeze(0).unsqueeze(0)

    print("input shapes:", hori.shape, vert.shape)
    model = Model()
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(np.concatenate((hori, vert), axis=1)))

    print("output shape:", output.shape)
    print(output)

