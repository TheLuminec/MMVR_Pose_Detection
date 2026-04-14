from dataset import create_dataloader
import torch
from torch.optim import Adam
from model import Model
import argparse

def train(root_path, batch_size, shuffle, epochs):
    print("Starting training...")
    dataloader = create_dataloader(root_path=root_path, batch_size=batch_size, shuffle=shuffle)
    model = Model()
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    print("Training...")
    for epoch in range(epochs):
        for batch in dataloader:
            radar, pose = batch
            output = model(radar)
            loss = criterion(output, pose)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='P1/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(args.root_path, args.batch_size, args.shuffle, args.epochs)
