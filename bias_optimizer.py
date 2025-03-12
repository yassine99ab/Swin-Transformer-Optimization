import numpy as np
import argparse
import tqdm
import random
import logging
from tqdm import tqdm
import model_driver
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torchvision import datasets, transforms
from torch.optim import AdamW, Adam
from torch.cuda.amp import autocast, GradScaler

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial

from PIL import ImageFilter, ImageOps, Image

from ignite.utils import convert_tensor

from hybrid_cmaes_with_de import HybridCMAESWithDE

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from genetic_algorithm_evolution import GeneticAlgorithmEvolution 
from utils.dataloader import dataload
from src.swin_vit import SwinTransformer
from utils.scheduler import build_scheduler  
from utils.dataloader import datainfo
from utils.optimizer import get_adam_optimizer
from utils.utils import clip_gradients
from utils.utils import save_checkpoint
from utils.cutmix import CutMix
from model_validator import ModelValidator
import warnings
warnings.filterwarnings("ignore")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser('SWIN ViT for CIFAR-10', add_help=False)
    parser.add_argument('--dir', type=str, default='./data',
                    help='Data directory')
    parser.add_argument('--num_classes', type=int, default=10, choices=[10, 100, 1000],
                    help='Dataset name')

    # Model parameters
    parser.add_argument('--patch_size', default=2, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=bool,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=32, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default=9, type=int, help=""" No. of layers of ViT """)
    parser.add_argument('--num_heads',default=12, type=int, help=""" No. of heads in attention layer
                                                                                    in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--drop_rate',default=0., type=float, help=""" dropout """)

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-1, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. Recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing for optimizer')
    parser.add_argument('--gamma', type=float, default=1.0,
                    help='Gamma value for Cosine LR schedule')

    # Misc
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'], help='Please specify path to the training data.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--mlp_head_in", default=192, type=int, help="input dimension going inside MLP projection head")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="directory to save checkpoints")

    args = parser.parse_args()
    logging.basicConfig(filename='bias_optimizer_cma_de_es.log', level=logging.INFO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")

    print("\n--- Downloading Data ---\n")

    data_info = datainfo(args)
    train_dataset, val_dataset = dataload(args, data_info)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    checkpoint = torch.load("experiments/model__hybrid_gen200_cma_run5_pop50.pt")


    # Step 2: Load model architecture (if necessary)
    # If the checkpoint contains only the state_dict:
    # Define your model architecture
    # model = MyModel()
    # model.load_state_dict(checkpoint)

    # If the checkpoint contains the entire model:
    model = checkpoint  # Assuming the saved file has the model itself
    # state_dict = model
    # Step 3: Inspect the keys in the state_dict to identify the last layer
    classification_head = model.head  # Change this based on your implementation

    # Print the head to verify its structure
    print(classification_head)

    # Access the bias parameter
    bias = classification_head.bias
    print(f"Bias before optimization: {bias}")
    print(type(bias))
    optimizer = Adam(
    [classification_head.bias],  # Optimizing only the bias
    lr=0.01,                     # Learning rate
    weight_decay=1e-2,           # Weight decay (L2 regularization)
    betas=(0.9, 0.999),          # Coefficients for moving averages of gradient and squared gradient
    eps=1e-8                     # Term added to denominator for numerical stability
    )
    criterion = nn.CrossEntropyLoss()
    # Convert model and inputs to float64
    model = model.float()
    
    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    min_delta = 0.01  # Minimum change in validation loss to qualify as improvement
    early_stop_counter = 0
    best_val_loss = float('inf')

    # Training and evaluation loop
    num_epochs = 50  # Maximum number of epochs
    model.to(device)  # Ensure the model is on the correct device
 # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 0.01  # Minimum change in validation loss to qualify as improvement
    early_stop_counter = 0
    best_val_loss = float('inf')

    # Training and evaluation loop
    num_epochs = 50
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()
                labels = labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        # Compute average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

       
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        torch.save(model, "experiments/optimized_bias_best_model.pt")
       

        # Load the best model for final evaluation
    model.load_state_dict(best_model_state)
    model.to(device)

    # Final evaluation on validation set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_val_accuracy = 100 * correct / total
    print(f"Final Validation Accuracy after Bias Optimization: {final_val_accuracy:.2f}%")

def compute_accuracy_with_bias_optimization(self, bias, data_loader):
    """
    Computes the accuracy of the model after updating the head.bias dynamically.

    Args:
        bias (torch.Tensor): The new bias values for the head layer.
        data_loader (torch.utils.data.DataLoader): The data loader for evaluation.

    Returns:
        float: The accuracy of the model on the given dataset.
    """
    # Set the bias dynamically
    self.model.head.bias.data = bias.to(self.device)

    # Set the model to evaluation mode
    self.model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move inputs and targets to the appropriate device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.float()
            labels = labels.long()

            # Forward pass
            outputs = self.model(inputs)

            # Get predictions and compute accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    # Return accuracy
    return 100.0 * correct / total if total > 0 else 0.0