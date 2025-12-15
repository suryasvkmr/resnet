import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader

from losses import FocalLoss

# Change these if you want slightly longer/shorter demo training
BATCHES_TO_RUN = 30
LR = 1e-3
BATCH_SIZE = 64

def get_digits_loader(train=True):
    # For “upgrade + demonstrate”, we just use the same Digits data.
    # (If you want a train/test split, we can add it, but this is enough for tonight.)
    digits = load_digits()
    X = torch.tensor(digits.images, dtype=torch.float32)  # (N,8,8)
    y = torch.tensor(digits.target, dtype=torch.long)

    X = X / 16.0
    X = X.unsqueeze(1)  # (N,1,8,8)
    X = F.interpolate(X, size=(224, 224))  # (N,1,224,224)
    X = X.repeat(1, 3, 1, 1)  # (N,3,224,224)

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train)

def train_one(loss_name: str, criterion: nn.Module, device: str, save_path: str):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = get_digits_loader(train=True)

    model.train()
    print(f"\n=== Training with {loss_name} ===")
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"[{loss_name}] Batch {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx >= BATCHES_TO_RUN:
            break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved {loss_name} model to: {save_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Train baseline and improved
    train_one(
        loss_name="CrossEntropy",
        criterion=nn.CrossEntropyLoss(),
        device=device,
        save_path="runs/ce.pth",
    )

    train_one(
        loss_name="FocalLoss",
        criterion=FocalLoss(gamma=2.0),
        device=device,
        save_path="runs/focal.pth",
    )

    print("\nDone training both loss variants.")

if __name__ == "__main__":
    main()
