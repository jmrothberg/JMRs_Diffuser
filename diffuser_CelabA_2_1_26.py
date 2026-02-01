"""
Conditional Diffusion Model for Image Generation
Author: Jonathan M. Rothberg
Supports: MNIST, CIFAR-10 (regular/optimized), and CelebA datasets
Features: U-Net architecture, self-attention, multi-GPU training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import argparse
import tempfile
import shutil
import sys

"""
Conditional Diffusion Model for Image Generation

Supported Datasets:
- MNIST: Handwritten digits (0-9), 28x28 grayscale
- CIFAR-10: 10 object classes, 32x32 RGB
- CIFAR-10 Optimized: Faster training with smaller model
- CelebA: Face generation, 64x64 RGB (unconditional)

CIFAR-10 Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Usage:
  Training:   python diffuser_CelabA_10_28_25.py --mode train
  Inference:  python diffuser_CelabA_10_28_25.py --mode inference

Features:
- Linear and cosine noise schedules
- Self-attention layers for improved quality
- Batch and sequential sample generation
- Automatic checkpoint saving and resumption
- Multi-GPU support with DataParallel
"""

class ConditionalUNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion models.

    Architecture:
    - Encoder: Downsamples features through convolutional layers
    - Middle: Bottleneck with optional self-attention
    - Decoder: Upsamples with skip connections from encoder

    Conditioning:
    - Time embeddings: Indicate noise level at current timestep
    - Class embeddings: Control generated class (MNIST digits, CIFAR-10 objects)
    - Both embeddings concatenated as additional input channels

    Args:
        num_classes: Number of classes for conditional generation
        emb_dim: Embedding dimension for time and class embeddings
        timesteps: Total diffusion timesteps
        in_channels: Input image channels (1 for grayscale, 3 for RGB)
        use_attention: Enable self-attention layers in bottleneck and decoder
        use_optimized_cifar10: Use smaller model variant for faster training
    """
    def __init__(self, num_classes=10, emb_dim=32, timesteps=1000, in_channels=3, use_attention=False, use_optimized_cifar10=False):
        super().__init__()

        # Enforce minimum embedding dimension for RGB images
        if in_channels == 3 and emb_dim < 128:
            emb_dim = 128

        # Configure model capacity based on dataset
        if use_optimized_cifar10 and in_channels == 3:
            # Optimized: Smaller model for faster CIFAR-10 training
            capacity_mult = 1.0
            base_init = 384
            base_down1 = 256
            base_down2 = 512
        else:
            # Standard: Larger capacity for better quality
            capacity_mult = 1.5 if in_channels == 3 else 1.0
            base_init = 512
            base_down1 = 1024
            base_down2 = 2048

        # Apply capacity multiplier to channel dimensions
        init_channels   = int(base_init * capacity_mult)
        down1_channels  = int(base_down1 * capacity_mult)
        down2_channels  = int(base_down2 * capacity_mult)
        up1_channels    = int(base_down1 * capacity_mult)
        up2_channels    = int(base_init * capacity_mult)
        final_ch1       = int(128 * capacity_mult)
        final_ch2       = int(64 * capacity_mult)
        
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.use_attention = use_attention  # Store this as model parameter
        self.use_optimized_cifar10 = use_optimized_cifar10  # Store optimized flag
        
        # Calculate total input channels including embeddings
        total_input_channels = in_channels + (2 * emb_dim)  # Original channels + time emb + class emb
        
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        self.time_embedding = nn.Embedding(timesteps, emb_dim)
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, init_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, init_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, init_channels),
            nn.SiLU(),
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(init_channels, down1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, down1_channels),
            nn.SiLU(),
            nn.Conv2d(down1_channels, down1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, down1_channels),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(down1_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        
        # Determine attention heads based on optimized CIFAR-10 setting
        attention_heads = 4 if use_optimized_cifar10 else 8

        middle_layers = []
        # First convolution block of the middle
        middle_layers.extend([
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
        ])
        if use_attention:
            middle_layers.append(SelfAttention(down2_channels, num_heads=attention_heads))
        middle_layers.extend([
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
        ])
        if use_attention:
            middle_layers.append(SelfAttention(down2_channels, num_heads=attention_heads))
        middle_layers.extend([
            nn.Conv2d(down2_channels, down2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, down2_channels),
            nn.SiLU(),
        ])
        self.middle = nn.Sequential(*middle_layers)
        
        up1_layers = [
            nn.Conv2d(down2_channels + down1_channels, up1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up1_channels),
            nn.SiLU(),
        ]
        if use_attention:
            up1_layers.append(SelfAttention(up1_channels, num_heads=attention_heads))
        up1_layers.extend([
            nn.Conv2d(up1_channels, up1_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up1_channels),
            nn.SiLU(),
        ])
        self.up1_conv = nn.Sequential(*up1_layers)

        up2_layers = [
            nn.Conv2d(up1_channels + init_channels, up2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up2_channels),
            nn.SiLU(),
        ]
        if use_attention:
            up2_layers.append(SelfAttention(up2_channels, num_heads=attention_heads))
        up2_layers.extend([
            nn.Conv2d(up2_channels, up2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, up2_channels),
            nn.SiLU(),
        ])
        self.up2_conv = nn.Sequential(*up2_layers)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        
        self.final = nn.Sequential(
            nn.Conv2d(up2_channels, final_ch1, kernel_size=3, padding=1),
            nn.GroupNorm(16, final_ch1),
            nn.SiLU(),
            nn.Conv2d(final_ch1, final_ch2, kernel_size=3, padding=1),
            nn.GroupNorm(16, final_ch2),
            nn.SiLU(),
            nn.Conv2d(final_ch2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c):
        """
        Forward pass to predict noise in input images.

        Args:
            x: Noisy images [batch, channels, height, width]
            t: Diffusion timesteps [batch]
            c: Class labels [batch]

        Returns:
            Predicted noise tensor matching input shape
        """
                
        # Clamp indices to prevent out-of-bounds access (DataParallel can cause issues)
        t = torch.clamp(t, 0, self.timesteps - 1)
        c = torch.clamp(c, 0, self.num_classes - 1)

        t_emb = self.time_embedding(t)
        c_emb = self.class_embedding(c)
        
        # Reshape embeddings to match image dimensions
        t_emb = t_emb.view(-1, self.emb_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        c_emb = c_emb.view(-1, self.emb_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenate along channel dimension
        x = torch.cat([x, t_emb, c_emb], dim=1)  # This will have in_channels + emb_dim + emb_dim channels
        
        # Down path with residual connections
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Middle with residual connection
        identity = x3
        x3 = self.middle(x3)
        x3 = x3 + identity  # Residual connection
        
        # Up path with skip connections
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1_conv(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2_conv(x)
        
        x = self.final(x)
        
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention for capturing long-range spatial dependencies.
    Improves generation of coherent global structures.

    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (8 for standard, 4 for optimized)
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, 4 * channels),  # Wider FF network
            nn.GELU(),
            nn.Linear(4 * channels, channels),  # Back to original channels
        )

    def forward(self, x):
        size = x.shape[-2:]  # Store original spatial dimensions
        
        # Reshape to sequence format and apply layer norm
        x = x.view(x.shape[0], self.channels, -1).swapaxes(1, 2)  # [B, HW, C]
        x_ln = self.ln(x)
        
        # Apply attention with residual connection
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        
        # Apply feedforward with residual
        attention_value = self.ff_self(attention_value) + attention_value
        
        # Reshape back to image format
        return attention_value.swapaxes(2, 1).view(x.shape[0], self.channels, *size)


class DiffusionModel:
    """
    DDPM (Denoising Diffusion Probabilistic Model) implementation.

    Forward Process:
        Gradually adds Gaussian noise to images over T timesteps until pure noise

    Reverse Process:
        Iteratively denoises from random noise using a trained neural network

    Noise Schedules:
        - Linear: Uniform noise increase (beta_start → beta_end)
        - Cosine: Smoother transitions, often better for complex images

    Args:
        timesteps: Number of diffusion steps (500-1000 typical)
        beta_start: Initial noise variance (1e-4 typical)
        beta_end: Final noise variance (0.01-0.02 typical)
        schedule_type: 'linear' or 'cosine'
        noise_scale: Sampling noise multiplier
        use_noise_scaling: If True, reduce noise gradually during sampling
        cosine_s: Cosine schedule offset parameter
        emb_dim: Embedding dimension for conditioning
        device: Device to use (cuda/mps/cpu) - if None, auto-detected
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear',
                 noise_scale=1.0, use_noise_scaling=False, cosine_s=0.008, emb_dim=128, device=None):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.noise_scale = noise_scale
        self.use_noise_scaling = use_noise_scaling
        self.cosine_s = cosine_s
        self.emb_dim = emb_dim

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device

        # Modify name_suffix to include embedding dimension
        self.name_suffix = f"ts{timesteps}_bs{beta_start:.0e}_be{beta_end:.0e}_emb{emb_dim}"

        # Linear or cosine beta schedule - device-agnostic
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(device)
        elif schedule_type == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps, s=self.cosine_s).to(device)
        else:
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Choose 'linear' or 'cosine'.")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine noise schedule from "Improved Denoising Diffusion Probabilistic Models"
        https://arxiv.org/abs/2102.09672

        Provides smoother noise transitions than linear schedule, often improving
        quality for complex images.

        Args:
            timesteps: Total diffusion steps
            s: Offset parameter controlling initial noise rate (0.005-0.01 typical)
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        # Transform timesteps using cosine function
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        # Normalize to ensure alpha_1 = 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Calculate beta values from alphas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clip betas to prevent numerical instability
        return torch.clip(betas, 0, 0.999)

    def add_noise(self, x, t):
        """Add Gaussian noise to images at specified timesteps using DDPM forward process."""
        noise = torch.randn_like(x)
        
        # Ensure proper broadcasting of alpha values
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        noisy_images = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha * noise
        return noisy_images, noise

    def sample(self, model, device, label, n_samples=1):
        """Generate images by iteratively denoising from random noise."""
        model.eval()
        # Fix: Properly access emb_dim when model is wrapped in DataParallel
        model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
        emb_dim = model_unwrapped.emb_dim  # read from unwrapped model
        
        with torch.no_grad():
            # The only line changed: subtract 2 * emb_dim instead of a fixed "32"
            in_channels = model_unwrapped.init_conv[0].weight.shape[1] - (2 * emb_dim)
            
            # The rest of your sample code is unchanged:
            # Handle different image sizes based on dataset
            if model_unwrapped.emb_dim >= 256:  # CelebA uses emb_dim=256
                image_size = 64
            elif in_channels == 3:  # CIFAR variants
                image_size = 32
            else:  # MNIST
                image_size = 28
            x = torch.randn(n_samples, in_channels, image_size, image_size).to(device)
            labels = torch.full((n_samples,), label, dtype=torch.long).to(device)
            
            for i in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), i, device=device, dtype=torch.long)
                
                predicted_noise = model(x, t, labels)
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                x_0_pred = (x - torch.sqrt(1. - alpha_cumprod) * predicted_noise) / \
                           torch.sqrt(alpha_cumprod)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)
                
                mean = (beta * x_0_pred + (1. - beta) * x) / torch.sqrt(alpha)
                
                if i > 0:
                    noise = torch.randn_like(x)
                    # Use use_noise_scaling parameter
                    if self.use_noise_scaling:
                        # Scale noise down over time
                        current_scale = self.noise_scale * (i / self.timesteps)
                        x = mean + torch.sqrt(beta) * noise * current_scale
                    else:
                        x = mean + torch.sqrt(beta) * noise * self.noise_scale
                else:
                    x = mean
            
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)
            return x

    def sample_batch(self, model, device, labels):
        """
        Generate multiple images in parallel for different class labels.
        Significantly faster than sequential generation.

        Args:
            model: Trained diffusion model
            device: CUDA device
            labels: List of class labels to generate

        Returns:
            List of generated image tensors
        """
        model.eval()
        # Fix: Properly access emb_dim when model is wrapped in DataParallel
        model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
        emb_dim = model_unwrapped.emb_dim  # read from unwrapped model

        num_samples = len(labels)

        with torch.no_grad():
            # The only line changed: subtract 2 * emb_dim instead of a fixed "32"
            in_channels = model_unwrapped.init_conv[0].weight.shape[1] - (2 * emb_dim)

            # The rest of your sample code is unchanged:
            # Handle different image sizes based on dataset
            if model_unwrapped.emb_dim >= 256:  # CelebA uses emb_dim=256
                image_size = 64
            elif in_channels == 3:  # CIFAR variants
                image_size = 32
            else:  # MNIST
                image_size = 28

            # Generate noise for all samples at once
            x = torch.randn(num_samples, in_channels, image_size, image_size).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

            for i in reversed(range(self.timesteps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)

                predicted_noise = model(x, t, labels_tensor)
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]

                x_0_pred = (x - torch.sqrt(1. - alpha_cumprod) * predicted_noise) / \
                           torch.sqrt(alpha_cumprod)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)

                mean = (beta * x_0_pred + (1. - beta) * x) / torch.sqrt(alpha)

                if i > 0:
                    noise = torch.randn_like(x)
                    # Use use_noise_scaling parameter
                    if self.use_noise_scaling:
                        # Scale noise down over time
                        current_scale = self.noise_scale * (i / self.timesteps)
                        x = mean + torch.sqrt(beta) * noise * current_scale
                    else:
                        x = mean + torch.sqrt(beta) * noise * self.noise_scale
                else:
                    x = mean

            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)

            # Return list of individual samples
            return [x[i:i+1] for i in range(num_samples)]

def save_samples(model, diffusion, device, epoch, avg_loss, dataset_name, batch_idx=None, use_batch_inference=True):
    """Generate and save image grid showing all classes or random samples."""
    # Check for DataParallel and access wrapped model properly
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model

    # Add attention to sample directory name
    attention_suffix = '_attention' if model_unwrapped.use_attention else ''
    sample_dir = f'samples_{dataset_name}_{diffusion.schedule_type}_{diffusion.name_suffix}{attention_suffix}'
    os.makedirs(sample_dir, exist_ok=True)

    # Generate samples for the grid
    if dataset_name == 'celeba':
        print("\nGenerating random faces for grid...")
        samples = []
        # For CelebA, generate 10 random faces (unconditional)
        for i in range(10):
            sample = diffusion.sample(model, device, 0, n_samples=1)  # Use class 0 (dummy)
            samples.append(sample)
    else:
        print("\nGenerating all digits...")
        samples = []

        if use_batch_inference:
            # NEW: Batch generation - generate all classes at once
            try:
                print("Using batch generation (10x faster)...")
                # Generate all 10 classes in one batch
                batch_samples = diffusion.sample_batch(model, device, list(range(10)))
                samples = [batch_samples[i] for i in range(10)]
                print("Batch generation successful!")
            except Exception as e:
                print(f"Batch generation failed ({e}), falling back to sequential...")
                use_batch_inference = False

        if not use_batch_inference:
            # ORIGINAL: Sequential generation (fallback)
            print("Using sequential generation...")
            for i in range(10):
                # Generate one sample at a time with same model parameters
                sample = diffusion.sample(model, device, i, n_samples=1)
                samples.append(sample)

    # Create and save the grid
    samples = torch.cat(samples, dim=0)
    grid = utils.make_grid(samples, nrow=5, normalize=True)

    # Save the grid
    filename = f'{sample_dir}/epoch_{epoch}_loss_{avg_loss:.4f}.png' if batch_idx is None else f'{sample_dir}/epoch_{epoch}_batch_{batch_idx}_loss_{avg_loss:.4f}.png'
    utils.save_image(grid, filename)
    method = "batch" if use_batch_inference else "sequential"
    print(f"\nSaved grid of all digits to {filename} (using {method} generation)")
    

def train(model, diffusion, dataloader, optimizer, device, num_epochs, dataset_name, override_lr=None, use_batch_inference=True):
    """
    Main training loop with automatic checkpointing and sample generation.
    Uses constant learning rate (proven best for DDPM) with optional plateau-based reduction.
    """
    model.train()

    # Use ReduceLROnPlateau - only reduces LR when loss stops improving
    # This is stable and proven for diffusion models
    # Most DDPM papers actually use constant LR, but this gives us automatic adjustment if needed
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by half when triggered
        patience=20,     # Wait 20 epochs of no improvement before reducing
        min_lr=1e-6      # Don't go below this
    )
    
    # Only set initial learning rate if explicitly overriding
    if override_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = override_lr
    
    # Rest of the existing train function remains exactly the same
    start_epoch = 0
    
    # Add attention to checkpoint directory name
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    attention_suffix = '_attention' if model_unwrapped.use_attention else ''
    checkpoint_dir = f'checkpoints_{dataset_name}_{diffusion.name_suffix}{attention_suffix}'
    checkpoint_path = None
    
    # Add attention to main checkpoint name
    main_checkpoint = f'diffusion_checkpoint_{dataset_name}_ts{diffusion.timesteps}_emb{diffusion.emb_dim}{attention_suffix}.pt'
    
    if os.path.exists(main_checkpoint):
        checkpoint = torch.load(main_checkpoint)
        if (checkpoint.get('dataset_name') == dataset_name and 
            checkpoint.get('emb_dim') == diffusion.emb_dim):
            checkpoint_path = main_checkpoint
            print(f"Found main checkpoint: {main_checkpoint}")
    else:
        # Modified pattern to include attention
        root_checkpoints = [f for f in os.listdir('.') 
                          if f.startswith(f'diffusion_checkpoint_{dataset_name}_ts')]
        if root_checkpoints:
            for ckpt_file in root_checkpoints:
                ckpt = torch.load(ckpt_file)
                if (ckpt.get('dataset_name') == dataset_name and
                    ckpt.get('emb_dim') == diffusion.emb_dim and
                    ckpt.get('use_attention', False) == model_unwrapped.use_attention and  # Check attention matches
                    ckpt.get('use_optimized_cifar10', False) == model_unwrapped.use_optimized_cifar10):  # Check optimized flag matches
                    checkpoint_path = ckpt_file
                    print(f"Found compatible checkpoint in root directory: {ckpt_file}")
                    break
        
        elif os.path.exists(checkpoint_dir):
            dir_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith(f'diffusion_checkpoint_{dataset_name}_ts')]
            if dir_checkpoints:
                dir_checkpoints.sort(key=lambda x: int(x.split('epoch_')[-1].split('.')[0]))
                latest_ckpt = torch.load(os.path.join(checkpoint_dir, dir_checkpoints[-1]))
                if (latest_ckpt.get('dataset_name') == dataset_name and
                    latest_ckpt.get('emb_dim') == diffusion.emb_dim and
                    latest_ckpt.get('use_attention', False) == model_unwrapped.use_attention and  # Check attention matches
                    latest_ckpt.get('use_optimized_cifar10', False) == model_unwrapped.use_optimized_cifar10):  # Check optimized flag matches
                    checkpoint_path = os.path.join(checkpoint_dir, dir_checkpoints[-1])
                    print(f"Found compatible checkpoint in directory: {checkpoint_path}")
    
    # Modified checkpoint loading logic
    if checkpoint_path:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Additional verification message
        print(f"Checkpoint details:")
        print(f"- Dataset: {checkpoint.get('dataset_name')}")
        print(f"- Embedding dim: {checkpoint.get('emb_dim')}")
        print(f"- Timesteps: {checkpoint.get('timesteps')}")
        print(f"- Using attention: {checkpoint.get('use_attention', False)}")
        
        # Check compatibility
        is_compatible = (
            checkpoint.get('timesteps') == diffusion.timesteps and
            checkpoint.get('emb_dim') == model_unwrapped.emb_dim and
            checkpoint.get('use_attention', False) == model_unwrapped.use_attention and
            checkpoint.get('use_optimized_cifar10', False) == model_unwrapped.use_optimized_cifar10
        )

        if not is_compatible:
            print("\nWARNING: Checkpoint parameters don't match current settings:")
            print(f"Checkpoint: timesteps={checkpoint.get('timesteps')}, "
                  f"emb_dim={checkpoint.get('emb_dim')}, "
                  f"attention={checkpoint.get('use_attention', False)}, "
                  f"optimized_cifar10={checkpoint.get('use_optimized_cifar10', False)}")
            print(f"Current: timesteps={diffusion.timesteps}, "
                  f"emb_dim={model_unwrapped.emb_dim}, "
                  f"attention={model_unwrapped.use_attention}, "
                  f"optimized_cifar10={model_unwrapped.use_optimized_cifar10}")
            print("\nCannot continue from this checkpoint due to architecture mismatch.")
            user_input = input("Start fresh training? (y/n): ")
            if user_input.lower() != 'y':
                print("Exiting...")
                sys.exit(0)
            checkpoint_path = None
        else:
            # Handle loading state dict for both DataParallel and non-DataParallel models
            state_dict = checkpoint['model_state_dict']
            
            # Check if we're loading a non-DataParallel checkpoint into a DataParallel model
            is_loading_non_parallel_to_parallel = (
                isinstance(model, nn.DataParallel) and
                not any(k.startswith('module.') for k in state_dict.keys())
            )
            
            if is_loading_non_parallel_to_parallel:
                print("Converting non-DataParallel checkpoint to DataParallel format...")
                # Create new state dict with 'module.' prefix for each key
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[f'module.{k}'] = v
                state_dict = new_state_dict
            
            # Check if we're loading a DataParallel checkpoint into a non-DataParallel model
            is_loading_parallel_to_non_parallel = (
                not isinstance(model, nn.DataParallel) and
                any(k.startswith('module.') for k in state_dict.keys())
            )
            
            if is_loading_parallel_to_non_parallel:
                print("Converting DataParallel checkpoint to non-DataParallel format...")
                # Create new state dict removing 'module.' prefix from each key
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                state_dict = new_state_dict
            
            # Now load the properly formatted state dict
            model.load_state_dict(state_dict)
            
            # Only load optimizer/scheduler state if not overriding learning rate
            if override_lr is None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print(f"Overriding learning rate to {override_lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = override_lr
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = f'checkpoints_{dataset_name}_{diffusion.name_suffix}{attention_suffix}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add this debug code at the start
    for images, _ in dataloader:
        print("Input image range:", images.min().item(), images.max().item())
        break
    
    # Modify loop to use start_epoch
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)

            # Handle unconditional datasets (like CelebA)
            if dataset_name == 'celeba':
                # For unconditional generation, use dummy labels (zeros)
                labels = torch.zeros(images.shape[0], dtype=torch.long, device=device)
            else:
                labels = labels.to(device)
            
            # Sample random timesteps for the entire batch
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)

            # Get noisy version of the entire batch
            noisy_images, target_noise = diffusion.add_noise(images, t)

            # Predict noise for the entire batch
            predicted_noise = model(noisy_images, t, labels)

            # Calculate loss for the entire batch at once
            loss = nn.MSELoss(reduction='mean')(predicted_noise, target_noise) # No loop needed

            optimizer.zero_grad()
            loss.backward()
            # Use higher gradient clipping for optimized CIFAR-10 model (smaller model needs less clipping)
            model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            max_norm = 1.0 if getattr(model_unwrapped, 'use_optimized_cifar10', False) else 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}')
                
                #if batch_idx % 500 == 0:
                #    save_samples(model, diffusion, device, epoch, batch_idx)
            
            # Inside training loop, add this debug print
            if batch_idx == 0 and epoch == 0:
                print("Predicted noise range:", predicted_noise.min().item(), predicted_noise.max().item())
                print("Target noise range:", target_noise.min().item(), target_noise.max().item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

        # Save latest checkpoint with diffusion parameters
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'timesteps': diffusion.timesteps,
            'schedule_type': diffusion.schedule_type,
            'dataset_name': dataset_name,  # Add dataset name to checkpoint
            'emb_dim': diffusion.emb_dim,  # Add emb_dim to checkpoint
            'use_attention': model_unwrapped.use_attention,  # Use unwrapped model here
            'use_optimized_cifar10': model_unwrapped.use_optimized_cifar10  # Add optimized flag
        }

        # Save main checkpoint - FIXED to include timesteps in filename
        torch.save(checkpoint, f'diffusion_checkpoint_{dataset_name}_ts{diffusion.timesteps}_emb{diffusion.emb_dim}{attention_suffix}.pt')
        print(f"Saved latest checkpoint at epoch {epoch}")

        # Save checkpoint based on dataset speed
        # Fast datasets (MNIST, CIFAR): every 10 epochs
        # Slow datasets (CelebA): every epoch
        should_save_checkpoint = (dataset_name == 'celeba') or (epoch % 10 == 0)

        if should_save_checkpoint:
            checkpoint_path = f'{checkpoint_dir}/diffusion_checkpoint_ts{diffusion.timesteps}_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}")

        # Step the scheduler with loss value (ReduceLROnPlateau needs this)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Manually print LR change (verbose parameter not available in all PyTorch versions)
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Only save samples once per epoch, at the end
        if epoch % 1 == 0:  # Every epoch
            save_samples(model, diffusion, device, epoch, avg_loss, dataset_name, use_batch_inference=use_batch_inference)

def inference_mode(model_path, device, dataset_name):
    """Interactive generation mode - input class labels to generate corresponding images."""
    # Look for any checkpoint file matching the dataset
    checkpoint_files = [f for f in os.listdir('.') if f.startswith(f'diffusion_checkpoint_{dataset_name}_ts')]
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found for {dataset_name}!")
        print("Please train the model first.")
        return
    
    # If multiple files exist, let user choose
    if len(checkpoint_files) > 1:
        print("\nAvailable checkpoints:")
        for idx, file in enumerate(checkpoint_files, 1):
            print(f"{idx}. {file}")
        while True:
            choice = input("\nSelect checkpoint number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(checkpoint_files):
                checkpoint_file = checkpoint_files[int(choice) - 1]
                break
            print("Invalid choice. Please try again.")
    else:
        checkpoint_file = checkpoint_files[0]
    
    print(f"\nLoading checkpoint: {checkpoint_file}")
    
    # Load checkpoint first to get parameters
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Extract emb_dim from checkpoint, default to 32 if not found
    emb_dim_ckpt = checkpoint.get('emb_dim', 32)
    print(f"Loaded emb_dim from checkpoint: {emb_dim_ckpt}") # Debug print

    # Create model with matching parameters from checkpoint
    in_channels = 3 if dataset_name.startswith('cifar10') or dataset_name == 'celeba' else 1  # Handle cifar10 variants and celeba
    # For CelebA, use unconditional generation (no class conditioning)
    num_classes = 1 if dataset_name == 'celeba' else 10
    model = ConditionalUNet(
        num_classes=num_classes,
        timesteps=checkpoint['timesteps'],  # Use timesteps from checkpoint
        in_channels=in_channels,
        emb_dim=emb_dim_ckpt, # Use loaded emb_dim here
        use_attention=checkpoint.get('use_attention', False), # Load use_attention from checkpoint, default to False if not found
        use_optimized_cifar10=checkpoint.get('use_optimized_cifar10', False) # Load optimized flag from checkpoint
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion model with matching parameters
    diffusion = DiffusionModel(
        timesteps=checkpoint['timesteps'],
        schedule_type=checkpoint.get('schedule_type', 'linear'),  # Default to linear if not found
        noise_scale=checkpoint.get('noise_scale', 1.0),
        use_noise_scaling=checkpoint.get('use_noise_scaling', False),
        cosine_s=checkpoint.get('cosine_s', 0.008),
        device=device
    )
    
    print("\nDiffusion Model Inference Mode")
    print("------------------------------")
    print("- Enter a digit (0-9) to generate an image")
    print("- Enter -1 to quit")
    print("- Generated images will be saved in 'inference_samples' directory")
    
    os.makedirs('inference_samples', exist_ok=True)
    
    while True:
        try:
            if dataset_name == 'celeba':
                # AUTOMATED: Generate grid of 10 random faces (unconditional)
                print("\nCelebA Face Generation:")
                print("- Generating grid of 10 random celebrity faces...")
                digit = 0  # Dummy value for unconditional
                # Skip to grid generation below
            elif dataset_name == 'cifar10':
                print("\nCIFAR10 classes:")
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
                for i, name in enumerate(classes):
                    print(f"{i}. {name}")
                digit = int(input("\nEnter class number (0-9) or -1 to quit: "))
            else:
                digit = int(input("\nEnter digit (0-9) or -1 to quit: "))

            if digit == -1 and dataset_name != 'celeba':
                print("Goodbye!")
                break
            if dataset_name != 'celeba' and (digit < 0 or digit > 9):
                print("Please enter a number between 0 and 9")
                continue

            if dataset_name == 'celeba':
                # Generate 10 random faces for grid
                print("Generating 10 random faces...")
                samples = []
                for i in range(10):
                    sample = diffusion.sample(model, device, 0)  # Dummy label
                    samples.append(sample)

                # Create grid
                samples_tensor = torch.cat(samples, dim=0)
                grid = utils.make_grid(samples_tensor, nrow=5, normalize=True, padding=2)

                # Save grid
                timestamp = torch.rand(1).item()
                filename = f'inference_samples/celeba_faces_{timestamp:.3f}.png'
                utils.save_image(grid, filename)
                print(f"✓ Saved face grid to {filename}")

                # Display grid for standalone inference
                plt.figure(figsize=(12, 6))
                grid_img = grid.permute(1, 2, 0).cpu()
                plt.imshow(grid_img)
                plt.axis('off')
                plt.title('Generated Celebrity Faces (2×5 grid)')
                plt.show()
                plt.close()

                print("\n✓ Generated grid of 10 celebrity faces!")
                continue  # Return to menu for more generation

            else:
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
                class_name = classes[digit] if dataset_name == 'cifar10' else str(digit)
                print(f"\nGenerating {class_name}...")

                # Generate image
                sample = diffusion.sample(model, device, digit)

                # Save image with appropriate name
                timestamp = torch.rand(1).item()
                filename = f'inference_samples/{class_name}_{timestamp:.3f}.png'
                utils.save_image(sample, filename, normalize=True)
                print(f"Saved to {filename}")

                # Display image
                plt.figure(figsize=(4, 4))  # Slightly larger for 64x64 CelebA images
                img = sample.squeeze().cpu()
                if dataset_name in ['cifar10', 'celeba']:
                    img = img.permute(1, 2, 0)  # CHW to HWC for RGB images
                plt.imshow(img, cmap=None if dataset_name in ['cifar10', 'celeba'] else 'gray')
                plt.axis('off')
                plt.title(f'Generated {class_name}')
                plt.show()
                #plt.pause(10)
                plt.close()
            
        except ValueError:
            if dataset_name == 'celeba':
                print("Error generating faces. Check model and try again.")
            else:
                print("Invalid input. Please enter a number between 0 and 9")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def get_input_with_default(prompt, default, convert_type=float):
    """Prompt user for input with default fallback value."""
    response = input(f"{prompt} (default={default}): ").strip()
    if response == "":
        return default
    try:
        return convert_type(response)
    except ValueError:
        if convert_type == str: # Handle string type directly
            return response if response else default
        print(f"Invalid input, using default: {default}")
        return default

def main():
    # Clear temp files at start
    temp_dir = tempfile.gettempdir()
    try:
        shutil.rmtree(os.path.join(temp_dir, 'torch_extensions'), ignore_errors=True)
    except:
        pass
    
    # Set device and random seed right at the start
    # Auto-detect best available device: CUDA (NVIDIA) > MPS (Apple) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA - {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon)")
        print("Note: MPS doesn't support multi-GPU. Using single device.")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU (WARNING: Training will be very slow!)")

    torch.manual_seed(42)
    
    try:
        # Set proper defaults in argument parser
        parser = argparse.ArgumentParser(description='MNIST/CIFAR Diffusion Model')
        parser.add_argument('--mode', type=str, default=None, choices=['train', 'inference'])
        parser.add_argument('--model_path', type=str, default='conditional_diffusion_mnist.pth')
        parser.add_argument('--timesteps', type=int, default=None)
        parser.add_argument('--beta_start', type=float, default=None)
        parser.add_argument('--beta_end', type=float, default=None)
        parser.add_argument('--epochs', type=int, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--noise_scale', type=float, default=None)
        parser.add_argument('--use_noise_scaling', action='store_true')
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--schedule_type', type=str, default=None, choices=['linear', 'cosine'])
        parser.add_argument('--cosine_s', type=float, default=None)
        parser.add_argument('--emb_dim', type=int, default=None, 
                           help='Embedding dimension (default: 128 for CIFAR, 32 for MNIST)')
        parser.add_argument('--use_attention', type=bool, default=None,
                           help='Use self-attention in the model')
        parser.add_argument('--num_gpus', type=int, default=None,
                           help='Number of GPUs to use (default: all available)')
        args = parser.parse_args()


        # Add dataset configuration right after device setup
        DATASETS = {
            'mnist': {
                'class': datasets.MNIST,
                'in_channels': 1,
                'image_size': 28,
                'normalize': ([0.5], [0.5]),
                # Add recommended defaults
                'defaults': {
                    'timesteps': 500,
                    'beta_start': 1e-5,
                    'beta_end': 0.01,
                    'batch_size': 128,
                    'learning_rate': 1e-3,
                    'schedule_type': 'linear',
                    'cosine_s': 0.005,
                    'noise_scale': 0.8,
                    'emb_dim': 32  # Default for MNIST
                }
            },
            'cifar10': {
                'class': datasets.CIFAR10,
                'in_channels': 3,
                'image_size': 32,
                'normalize': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                'defaults': {
                    'timesteps': 1000,          # Increased from 500 - CIFAR-10 needs more steps
                    'beta_start': 1e-4,         # Standard DDPM value
                    'beta_end': 0.02,           # Standard DDPM value for CIFAR-10
                    'batch_size': 128,          # Larger batch for more stable gradients
                    'learning_rate': 2e-4,      # Higher LR for faster convergence
                    'schedule_type': 'linear',  # Linear is proven for CIFAR-10
                    'cosine_s': 0.008,          # Not used with linear
                    'noise_scale': 1.0,         # Standard noise scale
                    'emb_dim': 128
                }
            },
            'cifar10_optimized': {
                'class': datasets.CIFAR10,
                'in_channels': 3,
                'image_size': 32,
                'normalize': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                'defaults': {
                    'timesteps': 1000,          # Increased from 500 for better quality
                    'beta_start': 1e-4,         # Standard DDPM value
                    'beta_end': 0.02,           # Standard DDPM value
                    'batch_size': 128,          # Large batch for smaller model
                    'learning_rate': 1e-4,      # LOWERED: Smaller model needs gentler LR (was 3e-4, caused mode collapse)
                    'schedule_type': 'linear',  # Linear schedule proven for CIFAR-10
                    'cosine_s': 0.008,          # Not used with linear
                    'noise_scale': 1.0,         # Standard noise scale
                    'emb_dim': 128,             # Keep embedding dimension
                    'use_optimized_cifar10': True  # Enable optimized model architecture
                }
            },
            'celeba': {
                'class': datasets.CelebA,
                'in_channels': 3,
                'image_size': 64,  # CelebA images are typically cropped to 64x64
                'normalize': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                'defaults': {
                    'timesteps': 1000,          # Optimal for high-resolution faces
                    'beta_start': 1e-4,         # Standard DDPM value
                    'beta_end': 0.02,           # Standard DDPM value
                    'batch_size': 16,           # Increased from 8 (still fits in 48GB VRAM)
                    'learning_rate': 2e-4,      # Increased from 1e-4 for faster convergence
                    'schedule_type': 'cosine',  # Cosine works better for complex images
                    'cosine_s': 0.008,          # Standard cosine parameter
                    'noise_scale': 1.0,         # Standard noise scale
                    'emb_dim': 256              # Large embedding for 64x64 resolution
                }
            }
        }

        if args.mode is None:
            print("\nDiffusion Model Training")
            print("1. Train model")
            print("2. Run inference")
            while True:
                mode_choice = input("\nEnter your choice (1 or 2): ").strip()
                if mode_choice == '1':
                    args.mode = 'train'
                    break
                elif mode_choice == '2':
                    args.mode = 'inference'
                    break
                print("Please enter 1 or 2")

        # Add dataset selection
        print("\nSelect dataset:")
        for idx, (name, info) in enumerate(DATASETS.items(), 1):
            if name == 'cifar10_optimized':
                print(f"{idx}. {name.upper()} ({info['image_size']}x{info['image_size']}, {info['in_channels']} channels) - FASTER, SINGLE GPU ONLY")
            elif name == 'cifar10':
                print(f"{idx}. {name.upper()} ({info['image_size']}x{info['image_size']}, {info['in_channels']} channels) - MULTI-GPU SUPPORTED")
            else:
                print(f"{idx}. {name.upper()} ({info['image_size']}x{info['image_size']}, {info['in_channels']} channels)")

        while True:
            dataset_choice = input("\nEnter your choice (1-{}): ".format(len(DATASETS))).strip()
            if dataset_choice.isdigit() and 1 <= int(dataset_choice) <= len(DATASETS):
                dataset_name = list(DATASETS.keys())[int(dataset_choice) - 1]
                dataset_config = DATASETS[dataset_name]
                break
            print(f"Please enter a number between 1 and {len(DATASETS)}")

        if args.mode == 'train':
            print("\n=== MNIST/CIFAR-10 Diffusion Model Parameter Guide ===")
            print("\nCore Parameters:")
            print(f"• Timesteps: {args.timesteps}")
            print("  - Higher values (1000+) = smoother transitions but slower training")
            print("  - Lower values (500-) = faster training but potentially lower quality")
            print("  - Recommended: 500 for CIFAR-10, 500-800 for MNIST")
            
            print(f"\n• Beta Schedule (start={args.beta_start}, end={args.beta_end})")
            print("  - Controls noise addition rate")
            print("  - Lower beta_start (1e-5) = clearer early steps")
            print("  - Higher beta_end (0.02-0.05) = more diverse samples")
            print("  - For MNIST: try beta_end=0.01-0.02")
            print("  - For CIFAR: try beta_end=0.02-0.05")
            
            print(f"\n• Noise Schedule Type: {args.schedule_type}")
            print("  - Linear: Simple uniform noise increase, good for starting out")
            print("  - Cosine: Often better quality but needs careful tuning")
            print("    • For MNIST: Use s=0.005 for gentler noise")
            print("    • For CIFAR: Use s=0.008-0.01 for complex images")

            # Get schedule type first
            if args.schedule_type is None:
                args.schedule_type = get_input_with_default("Noise Schedule (linear/cosine)", 'linear', str)
                
            # Only ask for cosine_s if cosine schedule is selected
            if args.schedule_type.lower() == 'cosine' and args.cosine_s is None:
                args.cosine_s = get_input_with_default("Cosine schedule offset (s)", 
                                                     0.008, 
                                                     float)
                print("  Note: Lower s (0.005) for simpler images like MNIST")
                print("        Higher s (0.008-0.01) for complex images like CIFAR")

            print(f"\n• Batch Size: {args.batch_size}")
            print("  - Larger = faster training but more memory")
            print("  - Smaller = more stable but slower")
            print("  - For MNIST: 64-128 usually works well")
            print("  - For CIFAR: 32-64 recommended")
            
            print(f"\n• Learning Rate: {args.learning_rate}")
            print("  - Higher (1e-3) = faster learning but potential instability")
            print("  - Lower (1e-5) = more stable but slower progress")
            print("  - With cosine schedule: try 1e-4 to 5e-4")
            print("  - With linear schedule: try 1e-3 to 5e-3")
            
            print("\nAdvanced Parameters:")
            print(f"• Noise Scale: {args.noise_scale}")
            print("  - Controls the amount of noise added during sampling")
            print("  - Higher (1.2+) = more diverse but potentially noisier samples")
            print("  - Lower (0.8-) = cleaner but less diverse samples")
            print(f"  - Default: {dataset_config['defaults']['noise_scale']} for {dataset_name.upper()}")
            print("  - Adjust if samples are too noisy or too similar")
            
            print(f"\n• Dynamic Noise Scaling: {'Enabled' if args.use_noise_scaling else 'Disabled'}")
            print("  - When enabled: noise scale decreases linearly from full value to 0")
            print("    • Starts at your chosen noise_scale value")
            print("    • Gradually reduces to 0 as sampling progresses")
            print("    • Formula: current_scale = noise_scale * (timestep / total_timesteps)")
            print("  - When disabled: noise scale stays constant at your chosen value")
            print("  - Default: Disabled")
            print("  - Enable for:")
            print("    • Potentially cleaner final results")
            print("    • More controlled noise reduction")
            print("  - Disable for:")
            print("    • Consistent noise level throughout")
            print("    • Traditional diffusion behavior")

            # Also update the input prompt to be clearer
            if args.use_noise_scaling is None:
                print("\nDynamic Noise Scaling:")
                print("- If enabled: noise scale will decrease linearly during sampling")
                print("- If disabled: noise scale will remain constant")
                use_scaling = input("Enable dynamic noise scaling? (y/n, default=y): ").lower().strip() or "y"
                args.use_noise_scaling = use_scaling == 'y'

            print(f"\n• Learning Rate: {args.learning_rate}")
            print("  - Higher (1e-3) = faster learning but potential instability")
            print("  - Lower (1e-5) = more stable but slower progress")
            print("  - With cosine schedule: try 1e-4 to 5e-4")
            print("  - With linear schedule: try 1e-3 to 5e-3")
            
            print("\nFor Other Datasets:")
            print("• Complex images (faces, natural scenes):")
            print("  - Increase timesteps (2000+)")
            print("  - Reduce learning rate (5e-5)")
            print("  - Consider larger model capacity")
            
            print("\n• Larger images:")
            print("  - Reduce batch size")
            print("  - Increase model capacity")
            print("  - Consider progressive training")
            
            print("\nMonitoring Tips:")
            print("• Watch for:")
            print("  - Loss not decreasing = try lower learning rate")
            print("  - Blurry samples = increase timesteps or model capacity")
            print("  - Training collapse = reduce learning rate or increase batch size")
            
            user_continue = input("\nPress Enter to continue with these settings, or 'q' to quit: ")
            if user_continue.lower() == 'q':
                print("Exiting...")
                return

            print("\nEnter parameters (press return to use defaults):")
            
            # When getting interactive input, use the same standard defaults
            if args.timesteps is None:
                args.timesteps = get_input_with_default("Timesteps", dataset_config['defaults']['timesteps'], int)
            if args.beta_start is None:
                args.beta_start = get_input_with_default("Beta start", dataset_config['defaults']['beta_start'])
            if args.beta_end is None:
                args.beta_end = get_input_with_default("Beta end", dataset_config['defaults']['beta_end'])
            if args.noise_scale is None:
                args.noise_scale = get_input_with_default("Noise scale", dataset_config['defaults']['noise_scale'])
                # Add clear noise scaling prompt right after noise scale input
                print("\nDynamic Noise Scaling:")
                print("  - When enabled: noise decreases gradually during sampling")
                print("  - When disabled: noise stays constant")
                use_scaling = input("Enable dynamic noise scaling? (y/n, default=y): ").lower().strip() or "y"
                args.use_noise_scaling = use_scaling == 'y'

            if args.schedule_type is None:
                args.schedule_type = get_input_with_default("Noise Schedule (linear/cosine)", dataset_config['defaults']['schedule_type'], str)
            if args.schedule_type.lower() == 'cosine' and args.cosine_s is None:
                args.cosine_s = get_input_with_default("Cosine schedule offset (s)", dataset_config['defaults']['cosine_s'])
            if args.epochs is None:
                args.epochs = get_input_with_default("Epochs", 500, int)
            if args.batch_size is None:
                args.batch_size = get_input_with_default("Batch size", dataset_config['defaults']['batch_size'], int)
            if args.learning_rate is None:
                args.learning_rate = get_input_with_default("Learning rate", dataset_config['defaults']['learning_rate'])
            if args.emb_dim is None:
                args.emb_dim = get_input_with_default("Embedding dimension", 
                                                    dataset_config['defaults'].get('emb_dim', 128), 
                                                    int)
                print(f"\nUsing embedding dimension: {args.emb_dim}")

            # Now check if attention is None before asking
            if args.use_attention is None:
                print("\nSelf-attention can improve image quality but increases training time and memory usage.")
                use_attention = input("Use self-attention in the model? (y/n, default=y): ").lower().strip() or "y"
                args.use_attention = use_attention == 'y'

            print(f"\nUsing parameters:")
            print(f"Timesteps: {args.timesteps}")
            print(f"Beta start: {args.beta_start}")
            print(f"Beta end: {args.beta_end}")
            print(f"Epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Noise Schedule: {args.schedule_type}")
            print(f"Learning rate: {args.learning_rate}")

            # Add GPU selection option (only for CUDA)
            if args.num_gpus is None:
                if device.type == 'cuda':
                    total_gpus = torch.cuda.device_count()
                    if total_gpus > 1:
                        print(f"\nMulti-GPU Training:")
                        print(f"Detected {total_gpus} CUDA GPUs available")
                        gpu_choices = [f"{i+1} GPU{'s' if i > 0 else ''}" for i in range(total_gpus)]
                        print("Options:")
                        for i, choice in enumerate(gpu_choices, 1):
                            print(f"{i}. Use {choice}")

                        while True:
                            gpu_choice = input(f"\nHow many GPUs to use? (1-{total_gpus}, default={total_gpus}): ").strip()
                            if gpu_choice == "":
                                args.num_gpus = total_gpus
                                break
                            elif gpu_choice.isdigit() and 1 <= int(gpu_choice) <= total_gpus:
                                args.num_gpus = int(gpu_choice)
                                break
                            print(f"Please enter a number between 1 and {total_gpus}")
                    else:
                        args.num_gpus = 1
                        print("\nSingle GPU training (only 1 CUDA GPU detected)")
                else:
                    # MPS or CPU - always single device
                    args.num_gpus = 1
                    print(f"\n{device.type.upper()} mode - single device")
            else:
                if device.type == 'cuda':
                    # Ensure num_gpus doesn't exceed available GPUs
                    args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
                    if args.num_gpus <= 0:
                        args.num_gpus = 1
                else:
                    # Force single device for MPS/CPU
                    args.num_gpus = 1

            if device.type == 'cuda':
                print(f"Using {args.num_gpus} GPU{'s' if args.num_gpus > 1 else ''}")
            else:
                print(f"Using {device.type.upper()} (single device)")

            # Ask about batch inference preference once (skip for CelebA)
            if dataset_name != 'celeba':
                print("\nSample Generation Options:")
                print("1. Batch generation (faster) - Generate all 10 classes at once")
                print("2. Sequential generation (slower, more stable) - Generate one class at a time")
                while True:
                    choice = input("\nChoose sample generation method (1 or 2, default=1): ").strip()
                    if choice == "" or choice == "1":
                        use_batch_inference = True
                        print("Using batch generation (faster)")
                        break
                    elif choice == "2":
                        use_batch_inference = False
                        print("Using sequential generation (slower but more stable)")
                        break
                    else:
                        print("Please enter 1 or 2")
            else:
                # For CelebA (unconditional), default to batch generation
                use_batch_inference = True
                print("Using batch generation for CelebA faces")

            # Modify dataset loading
            if dataset_name == 'celeba':
                # Special handling for CelebA - center crop and resize to 64x64
                transform = transforms.Compose([
                    transforms.CenterCrop(178),  # Center crop to square
                    transforms.Resize(64),       # Resize to 64x64
                    transforms.ToTensor(),
                    transforms.Normalize(*dataset_config['normalize'])
                ])
                # Check CelebA file structure
                print("Checking CelebA file structure...")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Looking for files in: {os.path.abspath('./data')}")

                # Check for required files
                img_dir = './data/img_align_celeba'
                partition_file = './data/list_eval_partition.txt'

                print(f"Checking: {os.path.abspath(img_dir)}")
                if os.path.exists(img_dir):
                    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
                    img_count = len(img_files)
                    print(f"✓ Found img_align_celeba/ with {img_count} images")
                    if img_count > 0:
                        print(f"  Sample files: {img_files[:3]}")
                else:
                    print(f"✗ Missing img_align_celeba/ directory")
                    print(f"  Expected at: {os.path.abspath(img_dir)}")

                print(f"Checking: {os.path.abspath(partition_file)}")
                if os.path.exists(partition_file):
                    print("✓ Found list_eval_partition.txt")
                    # Check if file is readable
                    try:
                        with open(partition_file, 'r') as f:
                            first_line = f.readline().strip()
                        print(f"  Sample content: {first_line}")
                    except Exception as e:
                        print(f"  Warning: Cannot read partition file: {e}")
                else:
                    print(f"✗ Missing list_eval_partition.txt")
                    print(f"  Expected at: {os.path.abspath(partition_file)}")

                if os.path.exists(img_dir) and os.path.exists(partition_file) and img_count > 0:
                    print("✓ CelebA files found - proceeding with training")
                    celeba_root = './data'
                else:
                    print("✗ CelebA dataset incomplete:")
                    if not os.path.exists(img_dir):
                        print("  - Missing: ./data/img_align_celeba/ directory")
                    if not os.path.exists(partition_file):
                        print("  - Missing: ./data/list_eval_partition.txt")
                    if img_count == 0:
                        print("  - No .jpg files found in img_align_celeba/")
                    print("\nPlease ensure files are in the correct locations.")
                    print("Current directory contents:")
                    try:
                        print(f"  ./data/: {os.listdir('./data')}")
                    except:
                        print("  Cannot list ./data/ contents")
                    raise FileNotFoundError("CelebA dataset files not found or incomplete")

                # Check for missing files in sequence
                print("Checking for missing files in CelebA sequence...")
                img_dir = './data/img_align_celeba'
                expected_files = {f"{i:06d}.jpg" for i in range(1, 202600)}  # 000001.jpg to 202599.jpg
                actual_files = set(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
                missing_files = expected_files - actual_files

                if missing_files:
                    print(f"✗ Missing {len(missing_files)} files from expected sequence")
                    print(f"  First few missing: {sorted(list(missing_files))[:10]}")
                    if len(missing_files) > 50:
                        print("  Too many missing files - this will cause PyTorch validation to fail")
                else:
                    print("✓ All expected files present in sequence")

                # Check partition file format
                print("Validating partition file format...")
                partition_file = './data/list_eval_partition.txt'
                valid_lines = 0
                invalid_lines = 0

                with open(partition_file, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) == 2 and parts[0].endswith('.jpg') and parts[1] in ['0', '1', '2']:
                            valid_lines += 1
                        else:
                            invalid_lines += 1
                            if invalid_lines <= 3:  # Show first few invalid lines
                                print(f"  Invalid line {i+1}: {line.strip()}")

                        if i >= 1000:  # Only check first 1000 lines for speed
                            break

                print(f"✓ Partition file: {valid_lines} valid lines, {invalid_lines} invalid lines")

                # Try manual dataset creation to bypass PyTorch validation
                print("Attempting manual CelebA dataset creation...")
                from torchvision.datasets import VisionDataset
                import PIL.Image as Image

                class SimpleCelebA(VisionDataset):
                    def __init__(self, root, transform=None):
                        super().__init__(root, transform=transform)
                        self.image_files = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])
                        print(f"Found {len(self.image_files)} images for manual dataset")

                    def __len__(self):
                        return len(self.image_files)

                    def __getitem__(self, idx):
                        img_path = os.path.join(self.root, self.image_files[idx])
                        image = Image.open(img_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        return image, 0  # Dummy label

                try:
                    print("Creating manual CelebA dataset...")
                    train_dataset = SimpleCelebA(
                        root='./data/img_align_celeba',
                        transform=transform
                    )
                    print(f"✓ Manual dataset created with {len(train_dataset)} images")
                except Exception as e:
                    print(f"✗ Manual dataset creation failed: {e}")
                    raise
            else:
                # Standard transforms for other datasets
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*dataset_config['normalize'])
                ])
                train_dataset = dataset_config['class'](
                    root='./data',
                    train=True,
                    transform=transform,
                    download=True
                )
            
            # Fix for DataParallel freezing: Reduce num_workers significantly
            # Too many workers can cause synchronization issues with DataParallel
            num_workers = min(4, torch.cuda.device_count() * 2) if torch.cuda.is_available() else 0
            print(f"Using {num_workers} DataLoader workers")

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                prefetch_factor=2 if num_workers > 0 else None  # Reduce prefetch to avoid memory issues
            )

            # Create diffusion model instance
            diffusion = DiffusionModel(
                timesteps=args.timesteps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                schedule_type=args.schedule_type,
                noise_scale=args.noise_scale,
                use_noise_scaling=args.use_noise_scaling,
                cosine_s=args.cosine_s,
                emb_dim=args.emb_dim,
                device=device
            )

            # Create model with correct channels and classes
            # For CelebA, use unconditional generation (no class conditioning)
            num_classes = 1 if dataset_name == 'celeba' else 10
            model = ConditionalUNet(
                num_classes=num_classes,
                emb_dim=args.emb_dim,
                timesteps=args.timesteps,
                in_channels=dataset_config['in_channels'],
                use_attention=args.use_attention,
                use_optimized_cifar10=dataset_config['defaults'].get('use_optimized_cifar10', False)
            ).to(device)
            
            # Multi-GPU setup with specific device selection
            # Note: MPS (Mac) and CPU don't support multi-GPU
            if device.type == 'mps':
                args.num_gpus = 1
                print("MPS (Apple Silicon) detected - multi-GPU not supported, using single device")
            elif device.type == 'cpu':
                args.num_gpus = 1
                print("CPU mode - multi-GPU not applicable")

            if args.num_gpus > 1 and device.type == 'cuda':
                # Check if this is the optimized CIFAR-10 model - force single GPU for stability
                model_unwrapped = model
                if getattr(model_unwrapped, 'use_optimized_cifar10', False):
                    print("\n" + "!"*60)
                    print("CIFAR-10 OPTIMIZED MODEL: Single GPU only")
                    print("!"*60)
                    print("The optimized model is designed for single-GPU training.")
                    print("Forcing num_gpus from {} to 1 for stability.".format(args.num_gpus))
                    print("Use standard CIFAR-10 (option 2) for multi-GPU support.")
                    print("!"*60 + "\n")
                    args.num_gpus = 1

                if args.num_gpus > 1:
                    print(f"\n{'='*60}")
                    print(f"Setting up DataParallel with {args.num_gpus} GPUs")
                    print(f"{'='*60}")
                    # Select specific devices
                    device_ids = list(range(args.num_gpus))
                    model = nn.DataParallel(model, device_ids=device_ids)
                    print(f"Model parallelized across GPUs: {device_ids}")
                    # Calculate and display effective batch size
                    effective_batch_size = args.batch_size * args.num_gpus
                    print(f"Effective batch size: {args.batch_size} × {args.num_gpus} = {effective_batch_size}")

                    # Add DataParallel-specific fixes for freezing issues
                    print("Applied DataParallel fixes:")
                    print("- Reduced DataLoader num_workers to prevent synchronization issues")
                    print("- Use persistent workers to avoid worker respawning overhead")
                    print("- Reduced prefetch factor to minimize memory pressure")
                    print("- If still freezing, try reducing batch_size or num_gpus")
                    print(f"{'='*60}\n")
                else:
                    print(f"\n{'='*60}")
                    print(f"Using SINGLE GPU (GPU 0) for training")
                    print(f"Batch size: {args.batch_size}")
                    print(f"{'='*60}\n")
            
            # Update optimizer settings - add learning rate scaling for multi-GPU
            base_lr = args.learning_rate
            if args.num_gpus > 1:
                # Optional: Linear scaling rule for multi-GPU
                # args.learning_rate = base_lr * args.num_gpus
                print(f"Note: Consider scaling learning rate for multi-GPU training")
                print(f"Current learning rate: {args.learning_rate}")
                scale_lr = input("Scale learning rate by number of GPUs? (y/n, default=n): ").lower().strip()
                if scale_lr == 'y':
                    args.learning_rate = base_lr * args.num_gpus
                    print(f"Scaled learning rate to: {args.learning_rate}")
            
            # Create optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.005,
                betas=(0.9, 0.999)
            )
            
            # Update checkpoint and model paths to include dataset name
            checkpoint_path = None
            # Fix: Access use_attention through module if using DataParallel
            model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            attention_suffix = '_attention' if model_unwrapped.use_attention else ''
            main_checkpoint = f'diffusion_checkpoint_{dataset_name}_ts{args.timesteps}_emb{args.emb_dim}{attention_suffix}.pt'

            print("\n" + "="*60)
            print("Starting Training Mode")
            print("="*60)

            print(f"Training on {len(train_dataset)} images")
            print(f"Batch size: {args.batch_size}")
            print(f"Epochs: {args.epochs}")
            print(f"\n💡 TIP: Press Ctrl-C to stop training safely")
            print("   - Saves emergency checkpoint")
            print("   - Cleans up GPU memory")
            print("   - Can resume from last checkpoint")
            print("="*60 + "\n")
            
            # Modified optimizer settings
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.005,              # Reduced from 0.01 to 0.005
                betas=(0.9, 0.999)              # Back to standard Adam betas
            )
            
            # Add override learning rate option when loading from checkpoint
            override_lr = None
            if os.path.exists(main_checkpoint):
                print("\nFound existing checkpoint.")
                override_choice = input("Would you like to override the learning rate? (y/n): ").lower()
                if override_choice == 'y':
                    override_lr = get_input_with_default("Enter new learning rate", args.learning_rate)
                    print(f"Will override learning rate to: {override_lr}")
                else:
                    print("Will continue with checkpoint's original learning rate")

            # When loading checkpoint, check embedding dimension match
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path)
                if checkpoint.get('emb_dim') != args.emb_dim:
                    print(f"\nWARNING: Checkpoint embedding dimension ({checkpoint.get('emb_dim')}) "
                          f"doesn't match requested dimension ({args.emb_dim})")
                    print("You'll need to start fresh training with the new embedding dimension.")
                    user_continue = input("Continue with fresh training? (y/n): ")
                    if user_continue.lower() != 'y':
                        print("Exiting...")
                        return

            # Set environment variables to help debug DataParallel issues
            if args.num_gpus > 1:
                # Only set CUDA_LAUNCH_BLOCKING if explicitly requested for debugging
                # TORCH_USE_CUDA_DSA=1 causes too many false positive warnings with complex models
                print("Using DataParallel with improved settings for stability")

            # Train the model
            try:
                train(model, diffusion, train_loader, optimizer, device, args.epochs, dataset_name, override_lr, use_batch_inference)
            except KeyboardInterrupt:
                print("\n" + "="*60)
                print("Training interrupted by user (Ctrl-C)")
                print("="*60)
                print("Cleaning up GPU memory...")

                # Clean up GPU memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Save emergency checkpoint
                model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
                attention_suffix = '_attention' if model_unwrapped.use_attention else ''
                emergency_checkpoint = f'diffusion_checkpoint_{dataset_name}_ts{args.timesteps}_emb{args.emb_dim}{attention_suffix}_INTERRUPTED.pt'

                print(f"Saving emergency checkpoint to: {emergency_checkpoint}")
                try:
                    checkpoint = {
                        'epoch': -1,  # Mark as interrupted
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'timesteps': args.timesteps,
                        'schedule_type': args.schedule_type,
                        'dataset_name': dataset_name,
                        'emb_dim': args.emb_dim,
                        'use_attention': model_unwrapped.use_attention,
                        'use_optimized_cifar10': getattr(model_unwrapped, 'use_optimized_cifar10', False),
                        'interrupted': True
                    }
                    torch.save(checkpoint, emergency_checkpoint)
                    print(f"✓ Emergency checkpoint saved successfully")
                except Exception as save_error:
                    print(f"✗ Failed to save emergency checkpoint: {save_error}")

                print("\nGPU cleanup complete. Safe to exit.")
                print("You can resume training from the last saved checkpoint.")
                return
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"Training error: {e}")
                print(f"{'='*60}")

                # Clean up GPU on error
                if device.type == 'cuda':
                    print("Cleaning up GPU memory...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if args.num_gpus > 1:
                    print(f"\nMulti-GPU troubleshooting:")
                    print(f"1. Reduce num_gpus from {args.num_gpus} to {max(1, args.num_gpus-1)}")
                    print(f"2. Reduce batch_size from {args.batch_size} to {max(1, args.batch_size//2)}")
                    print(f"3. Set CUDA_LAUNCH_BLOCKING=1 to debug exactly where it hangs")
                    print(f"4. Try single GPU training first to isolate the issue")
                raise e
            
            # Save final model
            torch.save(model.state_dict(), args.model_path)
            print(f"\nTraining complete! Model saved to {args.model_path}")
            
        else:  # inference mode
            try:
                inference_mode(args.model_path, device, dataset_name)
            except KeyboardInterrupt:
                print("\n\nInference interrupted by user (Ctrl-C)")
                print("Exiting gracefully...")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl-C)")
        print("Exiting...")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"An error occurred: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        if 'device' in locals() and torch.cuda.is_available():
            print("\nPerforming final GPU cleanup...")
            torch.cuda.empty_cache()
            if torch.cuda.device_count() > 0:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
            print("✓ GPU cleanup complete")

if __name__ == '__main__':
    main()