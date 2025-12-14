## Project Scope

This project improves an existing ResNet implementation from torchvision.
We do not re-run the full experimental study; instead, we upgrade and
demonstrate improvements to the training and evaluation pipeline.

## Improvements Implemented

1. **New Loss Function**
   - Replaced standard cross-entropy with focal loss to better handle
     class imbalance during training.

2. **Enhanced Evaluation Metrics**
   - Added per-class accuracy reporting and confusion matrix analysis
     to provide deeper insight into model behavior beyond overall accuracy.

These changes are modular and directly improve the original source code
without modifying the core torchvision implementation.
