# JMRs Diffuser

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for conditional and unconditional image generation.

## Author
Jonathan M. Rothberg

## Features

- **Multiple Dataset Support**
  - MNIST: Handwritten digits (28×28 grayscale)
  - CIFAR-10: Object classification (32×32 RGB)
  - CIFAR-10 Optimized: Faster training variant
  - CelebA: Face generation (64×64 RGB, unconditional)

- **Advanced Architecture**
  - U-Net with skip connections
  - Multi-head self-attention layers
  - Configurable embedding dimensions
  - Optimized model variants for faster training

- **Training Features**
  - Linear and cosine noise schedules
  - Multi-GPU support with DataParallel
  - Automatic checkpointing and resumption
  - OneCycleLR learning rate scheduling
  - Gradient clipping for stability
  - Batch and sequential sample generation

- **Inference**
  - Interactive generation mode
  - Class-conditional sampling
  - Batch generation for efficiency

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- matplotlib 3.5+
- numpy 1.20+
- Pillow 9.0+

## Usage

### Training

```bash
# Interactive mode (recommended for beginners)
python diffuser_CelabA_10_28_25.py --mode train

# Command line with parameters
python diffuser_CelabA_10_28_25.py \
  --mode train \
  --timesteps 500 \
  --beta_start 1e-5 \
  --beta_end 0.012 \
  --epochs 500 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --schedule_type linear \
  --emb_dim 128 \
  --use_attention True
```

### Inference

```bash
# Interactive generation
python diffuser_CelabA_10_28_25.py --mode inference
```

### Visualizing Training Loss

```bash
# GUI file picker
python diffuser_plot_loss_Oct_26_25.py

# Or specify directory directly
python diffuser_plot_loss_Oct_26_25.py /path/to/samples_directory
```

## Model Architecture

### Conditional U-Net

The model uses a U-Net architecture with the following components:

1. **Encoder**: Progressive downsampling through convolutional layers
2. **Bottleneck**: Optional self-attention for capturing global dependencies
3. **Decoder**: Upsampling with skip connections from encoder
4. **Conditioning**: Time and class embeddings concatenated as input channels

### Self-Attention

Multi-head self-attention layers capture long-range spatial dependencies, improving generation of coherent structures. Configurable number of attention heads (8 for standard model, 4 for optimized).

### Diffusion Process

**Forward Process**: Gradually adds Gaussian noise to images over T timesteps
```
x_t = √(α_t) * x_0 + √(1 - α_t) * ε
```

**Reverse Process**: Iteratively denoises from random noise using trained neural network
```
x_{t-1} = μ_θ(x_t, t) + σ_t * z
```

## Hyperparameters

### Recommended Settings

#### MNIST
- Timesteps: 500
- Beta schedule: Linear (1e-5 → 0.01)
- Batch size: 128
- Learning rate: 1e-3
- Embedding dimension: 32

#### CIFAR-10
- Timesteps: 500
- Beta schedule: Linear (1e-5 → 0.012)
- Batch size: 32
- Learning rate: 1e-4
- Embedding dimension: 128

#### CIFAR-10 Optimized
- Timesteps: 500
- Beta schedule: Linear (1e-5 → 0.012)
- Batch size: 128 (larger due to smaller model)
- Learning rate: 1e-4
- Embedding dimension: 128

#### CelebA
- Timesteps: 1000
- Beta schedule: Cosine (1e-4 → 0.02)
- Batch size: 8
- Learning rate: 1e-4
- Embedding dimension: 256

### Noise Schedules

**Linear**: Simple uniform increase from β_start to β_end
- Good for: Simple datasets (MNIST)
- Faster to tune

**Cosine**: Smoother transitions using cosine function
- Good for: Complex datasets (CelebA, CIFAR-10)
- Better quality but requires tuning offset parameter `s`

## Training Details

### Checkpointing

Models are automatically saved with the following naming convention:
```
diffusion_checkpoint_{dataset}_{timesteps}_{emb_dim}_{attention}.pt
```

Checkpoints include:
- Model weights
- Optimizer state
- Scheduler state
- Training epoch
- Loss history
- Configuration parameters

### Sample Generation

During training, sample grids are generated and saved to:
```
samples_{dataset}_{schedule}_{parameters}/
```

Filenames include epoch and loss for easy tracking:
```
epoch_50_loss_0.1234.png
```

### Multi-GPU Training

The model supports DataParallel for multi-GPU training:
- Automatically detected and configured
- Linear learning rate scaling option
- Gradient synchronization across GPUs
- Optimized DataLoader workers to prevent deadlocks

**Note**: Optimized CIFAR-10 model is recommended for single-GPU use for stability.

## Project Structure

```
Diffusers/
├── diffuser_CelabA_10_28_25.py          # Main script (all datasets)
├── diffuser_Multiparallel__attention5_Oct_25_25.py  # MNIST/CIFAR variant
├── diffuser_plot_loss_Oct_26_25.py     # Loss visualization utility
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── data/                                # Dataset storage (auto-downloaded)
├── checkpoints_*/                       # Training checkpoints
├── samples_*/                           # Generated samples during training
└── inference_samples/                   # Inference mode outputs
```

## CIFAR-10 Classes

0. Airplane
1. Automobile
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Tips for Best Results

### Training
- Start with recommended hyperparameters for your dataset
- Monitor loss curves - should decrease steadily
- Check sample quality every few epochs
- Use self-attention for better quality (trades off training time)
- Experiment with noise schedules if results are unsatisfactory

### Troubleshooting
- **Blurry samples**: Increase timesteps or model capacity
- **Training collapse**: Reduce learning rate or increase batch size
- **Out of memory**: Reduce batch size or use optimized model variant
- **NaN losses**: Lower learning rate, check for gradient explosions

### Performance
- Batch inference is ~10× faster than sequential for grids
- Multi-GPU training scales well for larger models
- Optimized CIFAR-10 variant trains 2-3× faster with similar quality

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Built with PyTorch and inspired by recent advances in diffusion models for image generation.
