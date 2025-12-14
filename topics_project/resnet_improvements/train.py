import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18

from losses import FocalLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Transforms -----
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    # ----- Fake Dataset (NO DOWNLOADS) -----
    trainset = datasets.FakeData(
        size=1000,
        image_size=(3, 224, 224),
        num_classes=10,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )

    # ----- Model (NO PRETRAINED WEIGHTS) -----
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    # ----- UPGRADE #1: Focal Loss -----
    criterion = FocalLoss(gamma=2.0)
    # baseline would be: nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----- Training Loop (SHORT DEMO RUN) -----
    model.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Stop early — demonstration only
        if batch_idx >= 10:
            break

    print("Training loop completed with Focal Loss.")


if __name__ == "__main__":
    main()
