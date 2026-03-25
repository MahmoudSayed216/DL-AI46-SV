
###======================================================================================================##
## == NOTE: PLEASE READ THIS PART FIRST =================================================================##
## ======================================================================================================##
## <HUMAN WRITTEN COMMENTS>                                                                              ##
## Testing the Neural Networks Training Golden Rule                                                      ##
##   Steps:                                                                                              ##
##     0 — Single-Sample Sanity Check (Overfitting on a single example)                                  ##
##     1 — Sanity Check (Overfitting on a small batch)                                                   ##
##     2 — Establish Baseline (Train simple model on full data) -> will hardly converge / the simplicity ##
##         of the model will fail to capture the complexity of the data                                  ##
##     3 — Reduce Bias / Fix Underfitting (Complex model) -> will learn, but will also overfit           ##
##     4 — Reduce Variance / Fix Overfitting (Regularization) -> learning + generalization               ##
##                                                                                                       ##
## PLEASE NOTE THAT THE FIRST ~133 lines of this file are all related to                                 ##
## random seed setting and environment configs,                                                          ##
## so I believe that no meaningfull comments are required here side, so i left the LLM's comments        ##
##                                                                                                       ##
## I WILL HAVE A NOTE FLAG BEFORE EVERY IMPORTANT PIECE OF COMMENT, SO YOU CAN DIRECTLY NAGIVATE TO      ##
## LINES WITH "NOTE: PLEASE READ THIS PART" BEFORE THEM with `CTRL+F`                                     ##
## The test will be conducted on the CIFAR-100 Dataset                                                   ##
## <END OF HUMAN WRITTEN COMMENTS>                                                                       ##
###======================================================================================================##

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works in all environments)
import matplotlib.pyplot as plt
import random
import os
import time


# =============================================================================
# 🔒 REPRODUCIBILITY — Fix all random seeds
# This is THE MOST IMPORTANT step for a production-grade experiment.
# Without this, results cannot be replicated across runs or machines.
# =============================================================================

SEED = 42

def set_seed(seed: int):
    """Fix all sources of randomness for full reproducibility."""
    random.seed(seed)                        # Python built-in RNG
    np.random.seed(seed)                     # NumPy RNG
    torch.manual_seed(seed)                  # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)             # PyTorch GPU RNG (single GPU)
    torch.cuda.manual_seed_all(seed)         # PyTorch GPU RNG (multi-GPU)
    # Ensures deterministic algorithms in cuDNN (slight speed cost, worth it)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# =============================================================================
# ⚙️ GLOBAL CONFIG — Single place to control all hyperparameters
# Centralizing config makes experiments easy to track and reproduce.
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100       # CIFAR-100 has 100 fine-grained classes
DATA_DIR = "/kaggle/working/"     # Where to download/cache CIFAR-100
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[CONFIG] Device      : {DEVICE}")
print(f"[CONFIG] Seed        : {SEED}")
print(f"[CONFIG] Num Classes : {NUM_CLASSES}")
print(f"[CONFIG] Output Dir  : {OUTPUT_DIR}")
print("-" * 60)


# =============================================================================
# 📦 DATA LOADING — Transforms & DataLoaders
# We define separate transforms for each phase:
#   - Basic (steps 0–2): just normalize, no augmentation
#   - Augmented (steps 3–4): add flips/crops to help generalization
# =============================================================================

# ImageNet-style normalization stats (also commonly used for CIFAR)
# These mean/std values are computed from the CIFAR-100 training set
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# --- Basic transform: normalize only (no augmentation) ---
# Used in early steps where we just want to verify the code is correct
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# --- Augmented transform: flip + crop + normalize ---
# Used in later steps to fight overfitting (regularization via data augmentation)
augmented_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),                     # 50% chance of horizontal mirror
    transforms.RandomCrop(32, padding=4),                  # Random crop with 4px padding
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color variation
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

def get_datasets(train_transform=basic_transform, test_transform=basic_transform):
    """Download CIFAR-100 and return train/test datasets."""
    train_set = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    return train_set, test_set

def get_loaders(train_set, test_set, batch_size: int, num_workers: int = 2):
    """Wrap datasets in DataLoaders for batched iteration."""
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader

# Download the dataset once up front (both train + test)
print("[DATA] Downloading / loading CIFAR-100 ...")
full_train_set, full_test_set = get_datasets()
print(f"[DATA] Train samples : {len(full_train_set)}")
print(f"[DATA] Test  samples : {len(full_test_set)}")
print("-" * 60)

##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>   
# Models Definitions
# We define two models:
#   1. SimpleCNN  — a relatively small model with few params [a little bit over 2 million params] → used in the first 3 steps of the golden rule
#      It basically stacks 2 convolution blocks [conv relu pool]
#   2. DeepCNN    — a relatively bigger and deeper model [around 5.3 million params] with BatchNorm and Dropout → used in the last 2 steps of the golden rule
#      Batchnorm is used to improve the training process and the gradients, Dropout is used to prevent
#      excessive overfitting, but it won't be enough, so regularization will eventually be used here
## <END OF HUMAN WRITTEN COMMENTS>   


class SimpleCNN(nn.Module):
    """
    A simple 3-layer CNN.
    Deliberately kept small so we can easily detect underfitting.
    Architecture: Conv → ReLU → MaxPool (x2) → FC → FC → Output
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 32×16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels (RGB)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # halve spatial dims
 
            # Block 2: 32×16×16 → 64×8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),   # 64 channels × 8×8 spatial
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
 
    def forward(self, x):
        return self.classifier(self.features(x))
 

class DeepCNN(nn.Module):
    """
    A deeper CNN with BatchNorm and Dropout.
    BatchNorm  → stabilizes training, allows higher LR
    Dropout    → regularization to combat overfitting (Step 4)
    Architecture: 3 Conv blocks + larger FC with dropout
    """
    def __init__(self, num_classes: int = 100, dropout_rate: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 64×16×16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),   # Normalize across batch for stable gradients
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 128×8×8 → 256×4×4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),   # Randomly zero out neurons during training
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters — useful for comparing model capacity."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 🛠️ TRAINING UTILITIES
# Reusable functions for training one epoch and evaluating accuracy.
# Keeping these separate from the step logic = clean, DRY code.
# =============================================================================

##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>   
## The basic training loop of pytorch
## - zeros out the gradients
## - performs a forward pass
## - computes the loss
## - computes the gradients using .backward()
## - updates the parametrs using optimizer.step()
## <END OF HUMAN WRITTEN COMMENTS>   

def train_one_epoch(model, loader, optimizer, criterion):
    """
    Run one full pass over the training data.
    Returns: average loss over all batches.
    """
    model.train()   # Enables Dropout, BatchNorm in training mode
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()          # Clear gradients from previous step
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights
        total_loss += loss.item()
    return total_loss / len(loader)



##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>   
## a function that evaluates the model over the test data
## it computes the accuracy and the average loss on the test data
## <END OF HUMAN WRITTEN COMMENTS>   
def evaluate(model, loader, criterion):
    """
    Evaluate the model on any DataLoader.
    Returns: (average loss, accuracy %)
    No gradient computation needed → saves memory and speeds up eval.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>   
## this function calls the train_one_epoch function, and then logs and stores 
## the per-epoch metrics for further examination.
## <END OF HUMAN WRITTEN COMMENTS>   
def run_training(
    model, train_loader, test_loader, optimizer, criterion,
    num_epochs: int, label: str, scheduler=None
):
    """
    Full training loop with epoch-level logging.
    Returns a history dict with losses and accuracies for plotting.
    """
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Params  : {count_params(model):,}")
    print(f"  Epochs  : {num_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if scheduler:
            scheduler.step()  # Adjust learning rate at end of epoch

        elapsed = time.time() - t0
        print(
            f"  Epoch [{epoch:02d}/{num_epochs}] "
            f"| Train Loss: {tr_loss:.4f} "
            f"| Test Loss: {te_loss:.4f} "
            f"| Test Acc: {te_acc:.2f}% "
            f"| Time: {elapsed:.1f}s"
        )

    return history


def plot_history(history: dict, title: str, filename: str):
    """
    Plot training/test loss curves and test accuracy, then save to disk.
    Visual inspection of curves is essential for diagnosing under/overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["test_loss"],  label="Test Loss",  color="tomato", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history["test_acc"], color="green", label="Test Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Test Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [PLOT] Saved → {save_path}")




##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>  
## Step 0 - Single-Sample Sanity Check (Overfitting on a single example)
## this step is used to verify that the model works, the loss decreases and that the model can generally
## map between the input and the output
## so, as its name suggests, it's a 'sanity check', to make sure that everything is fine [at least in theory].
## if this step doesn't work [the loss doesn't become 0], it means that something is wrong with the model/choice of optimizer/choice of loss function
## This step can save hours and hours of debugging
##_________________________________________________________
## results of running this step:
## after running this step, i verified that the model works without any errors
## and loss has actually become 0, so, this ensures that everything is conceptually and theoretically fine.
## NOTE you can take a look at lines 12-26 in the logs.txt file, it shows the results of this step
## <END OF HUMAN WRITTEN COMMENTS>  



print("\n" + "="*60)
print("🔎 STEP 0 — Single-Sample Sanity Check")
print("="*60)

# Pull exactly ONE sample from the dataset
single_image, single_label = full_train_set[0]
# Add batch dimension: (C, H, W) → (1, C, H, W) because PyTorch expects batches
single_image  = single_image.unsqueeze(0).to(DEVICE)
single_label_tensor = torch.tensor([single_label]).to(DEVICE)

print(f"  Image shape : {single_image.shape}")   # Should be torch.Size([1, 3, 32, 32])
print(f"  Label       : {single_label} (class index in 0–99)")

# Build a tiny model (SimpleCNN) and a simple DataLoader that repeats 1 sample
single_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
single_criterion = nn.CrossEntropyLoss()
single_optimizer = optim.Adam(single_model.parameters(), lr=1e-3)

# We'll create a minimal loader by repeating the one sample
single_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(single_image, single_label_tensor),
    batch_size=1, shuffle=False
)

# Train for many epochs — loss should reach near-zero (model memorizes the 1 sample)
step0_epochs = 200
step0_losses = []
single_model.train()
print(f"\n  Training on 1 sample for {step0_epochs} epochs ...")
for epoch in range(1, step0_epochs + 1):
    single_optimizer.zero_grad()
    output = single_model(single_image)
    loss   = single_criterion(output, single_label_tensor)
    loss.backward()
    single_optimizer.step()
    step0_losses.append(loss.item())

    if epoch % 50 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:03d}/{step0_epochs}] Loss: {loss.item():.6f}")

# ✅ EXPECTED: loss should be very close to 0.0 by epoch 200
final_loss = step0_losses[-1]
print(f"\n  Final loss on 1 sample: {final_loss:.6f}")
assert final_loss < 0.01, \
    f"❌ SANITY CHECK FAILED: model couldn't overfit 1 sample! Loss={final_loss:.4f}"
print("  ✅ PASSED — Model successfully memorized 1 sample (loss < 0.01)")

# Plot the loss curve for Step 0
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(step0_losses, color="purple", linewidth=1.5)
ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
ax.set_title("Step 0 — Single Sample Sanity Check (Loss Should → 0)")
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step0_single_sample.png"), dpi=120)
plt.close()
print("  [PLOT] Saved → outputs/step0_single_sample.png")



##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>  
## Step 1 — Sanity Check (Overfitting on a small batch)
## This step helps making further verification that the model is trainable over a small batch of data.
## If so, it probably means it can perform well on even bigger batches.
## It's also used for further sanity check.
## Additionally, This step should result in a very low training loss, and very high test loss]
## otherwise [especially if training loss doesn't become very low], it means that something is fundamentally wrong
##_________________________________________________________
## results of running this step:
## after running this step, i verified that the model works without any errors, 
## the training loss approached 0, the test loss was in fact very high [~20]
## so, we can rest assured that things are going in the right direction until now
## NOTE you can take a look at lines 29-142 in the logs.txt file, it shows the results of this step
## <END OF HUMAN WRITTEN COMMENTS>  


print("\n" + "="*60)
print("🟢 STEP 1 — Sanity Check (Overfit Small Batch)")
print("="*60)

STEP1_SAMPLES = 128   # Use only 128 training samples
STEP1_EPOCHS  = 100
STEP1_BS      = 32    # Small batch size

# Subset the training set to the first 128 samples
step1_subset = torch.utils.data.Subset(full_train_set, indices=list(range(STEP1_SAMPLES)))
step1_train_loader, step1_test_loader = get_loaders(
    step1_subset, full_test_set, batch_size=STEP1_BS
)
print(f"  Training on {STEP1_SAMPLES} samples, batch size {STEP1_BS}")

step1_model     = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
step1_criterion = nn.CrossEntropyLoss()
step1_optimizer = optim.Adam(step1_model.parameters(), lr=1e-3)

step1_history = run_training(
    step1_model, step1_train_loader, step1_test_loader,
    step1_optimizer, step1_criterion,
    num_epochs=STEP1_EPOCHS, label="Step 1 — Small Batch Sanity Check"
)

# ✅ EXPECTED: train loss should reach near 0 (model overfits the tiny set)
#             test accuracy will be low (expected — only 128 training samples!)
final_train_loss = step1_history["train_loss"][-1]
print(f"\n  Final train loss : {final_train_loss:.4f}")
print(f"  Final test  acc  : {step1_history['test_acc'][-1]:.2f}%")
if final_train_loss < 0.1:
    print("  ✅ PASSED — Model overfits small batch (train loss near 0)")
else:
    print("  ⚠️  WARNING — Model may be struggling to overfit. Check architecture/LR.")

plot_history(step1_history, "Step 1 — Small Batch Sanity Check", "step1_small_batch.png")

##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>  
## Step 2 - Establish Baseline (Train simple model on full data)
## the model will hardly converge / the simplicity of the model will fail to capture the
## complexity of the data                                 
## this step gives a baseline accuracy that's indeed unacceptable, but it's used to compare it with improved
## versions of the model and training methods
## you should be able to get a very low training loss.
## the test loss should decrease as well, but the model will still perform poorly on the testing data
## 
##_________________________________________________________
## results of running this step:
## after running this step, 
## the training loss decreased as well as the testing loss
## Both were high anyway but the model learned a few patterns
## the baseline accuracy reached by the model on the test set was 43.24% and it seemed like there was a small
## room for improvement, but it isn't worth it since the model started to plateau.
## NOTE you can take a look at lines 146-176 in the logs.txt file, it shows the results of this step
## <END OF HUMAN WRITTEN COMMENTS>  


print("\n" + "="*60)
print("🟡 STEP 2 — Establish Baseline (Full Data, Simple Model)")
print("="*60)

STEP2_EPOCHS = 20     # Enough to see convergence trend
STEP2_BS     = 128

step2_train_loader, step2_test_loader = get_loaders(
    full_train_set, full_test_set, batch_size=STEP2_BS
)

step2_model     = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
step2_criterion = nn.CrossEntropyLoss()
step2_optimizer = optim.SGD(
    step2_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
)
# StepLR: reduce LR by factor of 0.1 every 10 epochs (helps convergence)
step2_scheduler = optim.lr_scheduler.StepLR(step2_optimizer, step_size=10, gamma=0.1)

step2_history = run_training(
    step2_model, step2_train_loader, step2_test_loader,
    step2_optimizer, step2_criterion,
    num_epochs=STEP2_EPOCHS, label="Step 2 — Baseline", scheduler=step2_scheduler
)

baseline_acc = step2_history["test_acc"][-1]
print(f"\n  📊 Baseline Test Accuracy: {baseline_acc:.2f}%")
print("  (We will aim to beat this in Steps 3 and 4)")

plot_history(step2_history, "Step 2 — Baseline (Simple Model, Full Data)", "step2_baseline.png")



##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>  
## Step 3 — Reduce Bias / Fix Underfitting (Complex model) -> will learn, but will also overfit   
## in the previous step, the model underfitted, due to the lack of complexity in the model
## so, in this step, we will introduce a more complex and deeper model with a higher number of parameters
## to be able to extract better patterns/features from the input data
## the training strategy also improved in this step as we introduced batch norm and dropout for better gradients
## and minor resistance to overfitting. 
## and therefore, it should [in theory] perform better than the simple model.
## it's important to NOTE that if the model gets too complex, it may overfit
##_________________________________________________________
## results of running this step:
## the model in fact performed better than step 2 [59.63% accuracy], but it did overfit the training data as expected
## Improvement over baseline: +16.39%
## NOTE you can take a look at lines 180-219 in the logs.txt file, it shows the results of this step
## <END OF HUMAN WRITTEN COMMENTS>  

print("\n" + "="*60)
print("🔴 STEP 3 — Reduce Bias (Complex Model, No Regularization)")
print("="*60)

STEP3_EPOCHS = 25
STEP3_BS     = 128

# Re-use the same loaders (no augmentation yet — isolate the effect of model capacity)
step3_train_loader, step3_test_loader = get_loaders(
    full_train_set, full_test_set, batch_size=STEP3_BS
)

# DeepCNN: more layers, more channels, BatchNorm — but NO dropout yet
step3_model     = DeepCNN(num_classes=NUM_CLASSES, dropout_rate=0.0).to(DEVICE)
step3_criterion = nn.CrossEntropyLoss()
step3_optimizer = optim.SGD(
    step3_model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4
)
# CosineAnnealingLR: smoothly decays LR from initial to ~0 over T_max epochs
# Better than StepLR for complex models — avoids sudden LR drops
step3_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    step3_optimizer, T_max=STEP3_EPOCHS, eta_min=1e-4
)

print(f"  SimpleCNN params : {count_params(SimpleCNN()):,}")
print(f"  DeepCNN   params : {count_params(DeepCNN(dropout_rate=0.0)):,}")
print("  (More parameters = higher capacity = should reduce underfitting)")

step3_history = run_training(
    step3_model, step3_train_loader, step3_test_loader,
    step3_optimizer, step3_criterion,
    num_epochs=STEP3_EPOCHS, label="Step 3 — Complex Model (No Regularization)",
    scheduler=step3_scheduler
)

step3_acc = step3_history["test_acc"][-1]
print(f"\n  📊 Step 3 Test Accuracy : {step3_acc:.2f}%")
print(f"  📊 Baseline Accuracy    : {baseline_acc:.2f}%")
improvement = step3_acc - baseline_acc
print(f"  📈 Improvement over baseline: {improvement:+.2f}%")

plot_history(step3_history, "Step 3 — Reduce Bias (DeepCNN, No Regularization)", "step3_reduce_bias.png")


##=============================##
## NOTE: PLEASE READ THIS PART ##
##=============================##
## <HUMAN WRITTEN COMMENTS>  
## STEP 4 — Reduce Variance / Fix Overfitting (Regularization)
## To Address the issue of overfitting that was introduced in the previous model
## we can add a regularization term to the loss function [weight decay] to prevent the model from memorizing the training data
## additionally, we can add some data augmentation, to make the model immune to diverse types of variations in the input data
## this should result in a better test loss and accuracy that are close to the training loss and accuracy.
##_________________________________________________________
## results of running this step:
## the model in fact performed better in terms of test accuracy and loss [66.80% and 1.1606 test loss compared to 1.0595 train loss]
## which indicates that the problem of the overfitting was addressed successfully.
## the difference between the model trained in this step and the rest of the steps is that this model generalizes well to unseen data and 
##  has a better performance in terms of accuracy
## Final improvement over baseline: +23.56%
## NOTE you can take a look at lines 223-268 in the logs.txt file, it shows the results of this step
## <END OF HUMAN WRITTEN COMMENTS>  


# =============================================================================
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔵 STEP 4 — Reduce Variance / Fix Overfitting (Regularization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY: Step 3 likely shows that the complex model overfits (train loss <<
# test loss, test acc plateaus or drops). We now add regularization:
#
#   ✔ Dropout            — randomly zeroes neurons → forces redundant representations
#   ✔ Weight Decay (L2)  — penalizes large weights → smoother decision boundaries
#   ✔ Data Augmentation  — random flips/crops → model sees more variation → generalizes
#   ✔ Same LR schedule   — cosine annealing
#
# Expected behavior:
#   - Train/test loss gap should NARROW (gap was high in Step 3)
#   - Test accuracy should be HIGHER or more stable than Step 3
# =============================================================================
print("\n" + "="*60)
print("🔵 STEP 4 — Reduce Variance (DeepCNN + Regularization)")
print("="*60)

STEP4_EPOCHS    = 30
STEP4_BS        = 128
DROPOUT_RATE    = 0.5    # 50% dropout in fully-connected layers
WEIGHT_DECAY    = 5e-4   # L2 regularization strength

# Use augmented transforms for training (extra regularization via data diversity)
step4_aug_train_set, _ = get_datasets(
    train_transform=augmented_transform, test_transform=basic_transform
)
step4_train_loader, step4_test_loader = get_loaders(
    step4_aug_train_set, full_test_set, batch_size=STEP4_BS
)

print(f"  Dropout Rate   : {DROPOUT_RATE}")
print(f"  Weight Decay   : {WEIGHT_DECAY}")
print("  Data Augmentation : RandomHorizontalFlip + RandomCrop + ColorJitter")

# DeepCNN with dropout enabled this time
step4_model     = DeepCNN(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
step4_criterion = nn.CrossEntropyLoss()
step4_optimizer = optim.SGD(
    step4_model.parameters(), lr=0.05, momentum=0.9, weight_decay=WEIGHT_DECAY
)
step4_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    step4_optimizer, T_max=STEP4_EPOCHS, eta_min=1e-4
)

step4_history = run_training(
    step4_model, step4_train_loader, step4_test_loader,
    step4_optimizer, step4_criterion,
    num_epochs=STEP4_EPOCHS, label="Step 4 — Regularized DeepCNN",
    scheduler=step4_scheduler
)

step4_acc = step4_history["test_acc"][-1]
print(f"\n  📊 Step 4 Test Accuracy : {step4_acc:.2f}%")
print(f"  📊 Step 3 Test Accuracy : {step3_acc:.2f}%")
print(f"  📊 Baseline Accuracy    : {baseline_acc:.2f}%")
print(f"  📈 Final improvement over baseline: {step4_acc - baseline_acc:+.2f}%")

plot_history(step4_history, "Step 4 — Reduce Variance (DeepCNN + Regularization)", "step4_reduce_variance.png")


# =============================================================================
# 📊 FINAL SUMMARY — Compare all steps side by side
# =============================================================================
print("\n" + "="*60)
print("📊 FINAL RESULTS SUMMARY")
print("="*60)
print(f"  {'Step':<40} {'Test Acc':>10}")
print(f"  {'-'*50}")
print(f"  {'Step 0 — Single Sample (sanity, no test eval)':<40} {'N/A':>10}")
print(f"  {'Step 1 — Small Batch Sanity Check':<40} {step1_history['test_acc'][-1]:>9.2f}%")
print(f"  {'Step 2 — Baseline (SimpleCNN, full data)':<40} {baseline_acc:>9.2f}%")
print(f"  {'Step 3 — Reduce Bias (DeepCNN, no reg.)':<40} {step3_acc:>9.2f}%")
print(f"  {'Step 4 — Reduce Variance (DeepCNN + reg.)':<40} {step4_acc:>9.2f}%")
print("="*60)
print("\n  All plots saved to: ./outputs/")
print("  Done. ✅")