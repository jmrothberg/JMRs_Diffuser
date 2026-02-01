# CIFAR-10 Training Guide

## Why Your Model Was Failing

Your previous hyperparameters were too conservative for CIFAR-10's complexity:

### Previous Issues:
1. **Timesteps too low (500)** - CIFAR-10 needs 1000+ for good quality
2. **Beta values too conservative** - (1e-5 → 0.012) didn't add enough noise
3. **Batch size too small (32)** - Led to noisy gradients
4. **Learning rate too low (1e-4)** - Extremely slow convergence
5. **OneCycleLR scheduler** - Not ideal for diffusion models

## New Improved Settings

### CIFAR-10 Standard:
```python
timesteps: 1000        # Doubled for better quality
beta_start: 1e-4       # Standard DDPM value
beta_end: 0.02         # Standard DDPM value (67% increase)
batch_size: 128        # 4x larger for stable gradients
learning_rate: 2e-4    # 2x higher for faster learning
scheduler: CosineAnnealingWarmRestarts  # Better for diffusion
```

### CIFAR-10 Optimized (Recommended):
```python
timesteps: 1000        # Same quality, faster training
beta_start: 1e-4       # Standard value
beta_end: 0.02         # Standard value
batch_size: 128        # Large batch for small model
learning_rate: 3e-4    # Higher LR for optimized model
scheduler: CosineAnnealingWarmRestarts  # Gradual decay
```

## Expected Training Timeline

### With New Settings:
- **Epoch 10**: Basic shapes visible
- **Epoch 30**: Clear objects recognizable
- **Epoch 50**: Good quality, some fine details
- **Epoch 100**: High quality, sharp images
- **Epoch 200-300**: Near-perfect quality

### Memory Requirements:
- **Standard CIFAR-10**: ~10-12GB VRAM (single GPU)
- **Optimized CIFAR-10**: ~6-8GB VRAM (recommended)

## Quick Start Commands

### For Best Results (Optimized Model):
```bash
python diffuser_CelabA_10_28_25.py --mode train
# Select option 3: CIFAR10_OPTIMIZED
# Press Enter to use all defaults (they're now optimized!)
```

### Custom Training:
```bash
python diffuser_CelabA_10_28_25.py \
  --mode train \
  --timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 0.02 \
  --epochs 300 \
  --batch_size 128 \
  --learning_rate 3e-4 \
  --emb_dim 128 \
  --use_attention True
```

## Monitoring Training

### What to Look For:

#### Loss Curves:
- Should decrease steadily from ~0.5 to ~0.05-0.1
- If loss plateaus above 0.2, increase learning rate
- If loss is erratic, reduce learning rate or increase batch size

#### Sample Quality:
- **Bad sign**: Fuzzy blobs after 50 epochs → increase timesteps or beta_end
- **Good sign**: Clear object shapes by epoch 30
- **Great sign**: Fine details like wheels, wings by epoch 100

#### Training Speed:
- **Standard**: ~3-5 min/epoch on RTX 3090
- **Optimized**: ~1-2 min/epoch on RTX 3090
- Should generate 10 samples at end of each epoch

## Troubleshooting

### Problem: Still getting blurry images after 100 epochs

**Solutions:**
1. Check you're using the NEW defaults (timesteps=1000, beta_end=0.02)
2. Verify self-attention is enabled: `--use_attention True`
3. Increase embedding dimension: `--emb_dim 256`
4. Try cosine schedule: `--schedule_type cosine --cosine_s 0.008`

### Problem: Training is too slow

**Solutions:**
1. Use CIFAR-10 Optimized (2-3x faster, similar quality)
2. Reduce batch size if GPU memory is limiting: `--batch_size 64`
3. Disable attention for speed (quality will suffer): `--use_attention False`

### Problem: Model diverges (NaN loss)

**Solutions:**
1. Reduce learning rate: `--learning_rate 1e-4`
2. Enable gradient clipping (already enabled at 0.5)
3. Reduce batch size: `--batch_size 64`
4. Check GPU memory isn't exhausted

### Problem: Out of memory

**Solutions:**
1. Use CIFAR-10 Optimized instead of standard
2. Reduce batch size: `--batch_size 64` or `--batch_size 32`
3. Use single GPU: `--num_gpus 1`
4. Disable attention: `--use_attention False`

## Advanced Tips

### For Absolute Best Quality:
```python
timesteps: 2000         # Even more gradual noise
beta_start: 1e-4
beta_end: 0.02
batch_size: 256         # Very stable training
learning_rate: 2e-4
emb_dim: 256            # More capacity
schedule_type: cosine   # Smoother noise schedule
cosine_s: 0.008
```

### For Fastest Training (acceptable quality):
```python
timesteps: 500          # Faster sampling
beta_start: 1e-4
beta_end: 0.02
batch_size: 256         # Compensate with large batch
learning_rate: 5e-4     # Aggressive learning
use_optimized_cifar10: True
use_attention: False    # Skip attention
```

## Comparing to Your Old Settings

| Setting | Old (Bad) | New (Good) | Impact |
|---------|-----------|------------|--------|
| Timesteps | 500 | 1000 | 2x better denoising |
| Beta End | 0.012 | 0.02 | 67% more noise coverage |
| Batch Size | 32 | 128 | 4x more stable gradients |
| Learning Rate | 1e-4 | 2e-4 | 2x faster convergence |
| Scheduler | OneCycleLR | CosineWarmRestarts | Better for diffusion |

## Expected Results

With the new settings, you should see:

### After 50 epochs:
- Clear object boundaries
- Correct colors
- Recognizable classes (airplanes have wings, cars have wheels)

### After 100 epochs:
- Sharp edges
- Fine details visible
- Realistic textures
- All 10 classes generated correctly

### After 200 epochs:
- Near-photorealistic quality
- Subtle lighting effects
- Diverse variations within each class

## Comparison to Other Models

Your model with these settings should achieve:
- **FID Score**: ~20-30 (after 200 epochs)
- **Quality**: Comparable to standard DDPM
- **Training Time**: ~10-20 hours for 200 epochs (single RTX 3090)

## References

These settings are based on:
- Original DDPM paper (Ho et al., 2020)
- Improved DDPM (Nichol & Dhariwal, 2021)
- Community best practices for CIFAR-10

## Need Help?

If you're still getting poor results after following this guide:
1. Check your generated samples in `samples_cifar10_*/`
2. Plot your loss curve: `python diffuser_plot_loss_Oct_26_25.py`
3. Verify timesteps in checkpoint filename (should be ts1000, not ts500)
4. Ensure you deleted old checkpoints before retraining
