import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 64

def get_digits_loader():
    digits = load_digits()
    X = torch.tensor(digits.images, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)

    X = X / 16.0
    X = X.unsqueeze(1)
    X = F.interpolate(X, size=(224, 224))
    X = X.repeat(1, 3, 1, 1)

    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

def load_model(weights_path: str, device: str):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def eval_model(name: str, model, loader, device: str):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n==============================")
    print(f"Evaluation: {name}")
    print(f"==============================")

    # Core metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # --- Extra metrics (easy + meaningful) ---
    acc = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\nExtra Metrics:")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loader = get_digits_loader()

    # Evaluate both saved models
    ce_model = load_model("runs/ce.pth", device)
    focal_model = load_model("runs/focal.pth", device)

    eval_model("CrossEntropy", ce_model, loader, device)
    eval_model("FocalLoss", focal_model, loader, device)

if __name__ == "__main__":
    main()
