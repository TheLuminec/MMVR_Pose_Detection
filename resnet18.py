import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18

class ResNet18PoseBaseline(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()

        self.model = resnet18(weights=None)

        # Original ResNet expects 3 input channels.
        # MMVR radar has 2 channels, so change first conv layer.
        self.model.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace classification layer with pose regression output.
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_keypoints * 2
        )

        self.num_keypoints = num_keypoints

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, self.num_keypoints, 2)

'''
def train_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MMVRPoseDataset("../test_example")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ResNet18PoseBaseline().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 50

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for radar, keypoints in train_loader:
            radar = radar.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()

            preds = model(radar)
            loss = criterion(preds, keypoints)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_pck = 0
        val_mae = 0

        with torch.no_grad():
            for radar, keypoints in val_loader:
                radar = radar.to(device)
                keypoints = keypoints.to(device)

                preds = model(radar)

                loss = criterion(preds, keypoints)
                val_loss += loss.item()

                val_pck += calculate_pck(preds, keypoints)
                val_mae += torch.mean(torch.abs(preds - keypoints)).item()

        val_loss /= len(val_loader)
        val_pck /= len(val_loader)
        val_mae /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PCK: {val_pck:.4f} | "
            f"MAE: {val_mae:.4f}"
        )

    torch.save(model.state_dict(), "resnet18_baseline.pth")
    print("Saved ResNet baseline model as resnet18_baseline.pth")


if __name__ == "__main__":
    train_resnet()
'''