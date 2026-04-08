from model import Model
import torch
import numpy as np

if __name__ == "__main__":
    data = np.load('test_example/00383_radar.npz')
    hori = data['hm_hori']  # (256, 128)
    vert = data['hm_vert']  # (256, 128)
    print("raw NaNs:", np.isnan(hori).any(), np.isnan(vert).any())

    # Conv2d expects tensors shaped as [batch, channel, height, width].
    hori = torch.from_numpy(hori).float().nan_to_num(0.0).unsqueeze(0).unsqueeze(0)
    vert = torch.from_numpy(vert).float().nan_to_num(0.0).unsqueeze(0).unsqueeze(0)

    print("input shapes:", hori.shape, vert.shape)
    model = Model()
    model.eval()

    with torch.no_grad():
        output = model(hori, vert)

    print("output shape:", output.shape)
    print(output)

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # dataset = None
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 
    # for epoch in range(10):
    #     for inputs, targets in dataloader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
