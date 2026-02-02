import os
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def list_sample_folders():
    """List all sample folders in current directory and let user select one."""
    # Find all folders starting with 'samples_'
    sample_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('samples_')]
    
    if not sample_dirs:
        print("No sample folders found (looking for folders starting with 'samples_')")
        return None
    
    # Sort by modification time (most recent first)
    sample_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print("\n=== Available Sample Folders ===")
    for i, folder in enumerate(sample_dirs, 1):
        # Count PNG files and get epoch range
        pngs = [f for f in os.listdir(folder) if f.endswith('.png')]
        epochs = []
        for f in pngs:
            match = re.search(r'epoch_(\d+)_', f)
            if match:
                epochs.append(int(match.group(1)))
        
        epoch_info = f"epochs {min(epochs)}-{max(epochs)}" if epochs else "no epochs"
        print(f"  {i}. {folder} ({len(pngs)} images, {epoch_info})")
    
    print(f"  0. Enter custom path")
    
    while True:
        choice = input(f"\nSelect folder (1-{len(sample_dirs)}, or 0 for custom): ").strip()
        if choice == "0":
            custom = input("Enter path: ").strip()
            if os.path.isdir(custom):
                return custom
            print("Invalid directory")
        elif choice.isdigit() and 1 <= int(choice) <= len(sample_dirs):
            return sample_dirs[int(choice) - 1]
        else:
            print(f"Please enter 0-{len(sample_dirs)}")

def select_directory_gui():
    """Open file dialog to select directory containing loss plot PNG files."""
    try:
        from tkinter import filedialog
        # Use native OS file dialog - let user pick any directory
        selected_dir = filedialog.askdirectory(
            title="Select Directory Containing Loss Plot Images",
            initialdir=os.getcwd()
        )

        if not selected_dir:
            return None

        # Return the full path
        return selected_dir
    except Exception as e:
        print(f"GUI not available (running headless): {e}")
        return None

def plot_loss_from_filenames(directory=None):
    """
    Extract and plot training loss from sample image filenames.

    Creates dual-panel visualization:
    - Left: Raw loss values per epoch
    - Right: 10-epoch sliding window average

    Expected filename format: epoch_X_loss_Y.png
    """
    # Select directory: command line arg > interactive list > GUI fallback
    if directory is None:
        selected_dir = list_sample_folders()
        if not selected_dir:
            # Fallback to GUI if no sample folders found
            selected_dir = select_directory_gui()
        if not selected_dir:
            print("No directory selected. Usage: python diffuser_plot_loss_Oct_26_25.py [directory_path]")
            return
    else:
        selected_dir = directory

    print(f"Processing directory: {selected_dir}")

    # Get all PNG files in the directory
    files = [f for f in os.listdir(selected_dir) if f.endswith('.png')]

    if not files:
        print(f"No PNG files found in {selected_dir}")
        return

    # Extract epoch and loss values using regex
    data = []
    for filename in files:
        match = re.search(r'epoch_(\d+)_loss_([\d.]+)\.png', filename)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            data.append((epoch, loss))

    if not data:
        print(f"No valid loss values found in {selected_dir}")
        return

    # Sort by epoch
    data.sort(key=lambda x: x[0])
    epochs, losses = zip(*data)
    losses = np.array(losses)
    
    # Print summary stats
    print(f"\n=== Loss Summary ===")
    print(f"Epochs: {min(epochs)} to {max(epochs)} ({len(epochs)} total)")
    print(f"Loss range: {losses.min():.6f} to {losses.max():.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {losses.min():.6f} (epoch {epochs[np.argmin(losses)]})")

    # Create short name for directory (remove 'samples_' prefix and common parts)
    dir_basename = os.path.basename(selected_dir)
    short_name = dir_basename.replace('samples_', '').replace('_linear_', '_').replace('_attention', '')

    # Calculate sliding window average (last 10)
    window_size = 10
    sliding_avg = []
    if len(losses) >= window_size:
        for i in range(len(losses)):
            start_idx = max(0, i - window_size + 1)
            window = losses[start_idx:i+1]
            sliding_avg.append(np.mean(window))
    else:
        sliding_avg = losses.copy()  # Fallback if not enough data

    # ===== SINGLE FIGURE WITH TWO SUBPLOTS =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ===== LEFT PLOT: Raw Loss =====
    ax1.plot(epochs, losses, 'b-', linewidth=1)
    ax1.scatter(epochs, losses, c='blue', alpha=0.5, s=20)

    ax1.set_title(f'Raw Loss - {short_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # ===== RIGHT PLOT: Sliding Window Average =====
    if len(losses) >= window_size:
        ax2.plot(epochs, sliding_avg, 'g-', linewidth=2)
        ax2.scatter(epochs, losses, c='lightgray', alpha=0.3, s=10)
    else:
        # Not enough data for sliding window
        ax2.plot(epochs, losses, 'g-', linewidth=1)
        ax2.scatter(epochs, losses, c='green', alpha=0.5, s=20)

    ax2.set_title(f'Sliding Window Average (Window={window_size}) - {short_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save combined plot with date stamp to avoid overwriting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'loss_combined_{short_name}_{timestamp}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Combined plot saved as: {plot_filename}")

    # Try to show the plot interactively
    try:
        plt.show()
        print("Plot displayed in window. Close the window to continue.")
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print("Plot was saved to file instead.")

    # Close the plot to free memory
    plt.close(fig)

if __name__ == '__main__':
    # Check for command line argument
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        if os.path.isdir(directory):
            plot_loss_from_filenames(directory)
        else:
            print(f"Error: {directory} is not a valid directory")
    else:
        plot_loss_from_filenames() 