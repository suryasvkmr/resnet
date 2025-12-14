import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Transforms -----
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    # ----- Fake Dataset (NO DOWNLOADS) -----
    testset = datasets.FakeData(
        size=500,
        image_size=(3, 224, 224),
        num_classes=10,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False
    )

    # ----- Model (NO PRETRAINED WEIGHTS) -----
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            # Short evaluation only
            if len(all_preds) >= 500:
                break

    # ----- UPGRADE #2: Enhanced Evaluation Metrics -----
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
