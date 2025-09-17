"""
Utility functions for MicroMedGPT
Includes plotting, metrics, and other helper functions
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from config import *

def plot_training_curves(eval_iterations, train_losses, val_losses, learning_rates, grad_norms):
    """
    Plot training and validation curves
    
    Args:
        eval_iterations: list of iteration numbers when losses were evaluated
        train_losses: list of training losses
        val_losses: list of validation losses
        learning_rates: list of learning rates over training
        grad_norms: list of gradient norms over training
    """
    
    # Plot training & validation loss
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(eval_iterations, train_losses, label='Train Loss', color='blue')
    plt.plot(eval_iterations, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate schedule
    plt.subplot(1, 3, 2)
    plt.plot(range(len(learning_rates)), learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Cosine Annealing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot gradient norms
    plt.subplot(1, 3, 3)
    plt.plot(range(len(grad_norms)), grad_norms, label='Gradient Norm', color='purple')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Clipping Threshold')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_loss_only(eval_iterations, train_losses, val_losses):
    """Plot only the loss curves"""
    plt.figure(figsize=(8, 5))
    plt.plot(eval_iterations, train_losses, label='Train Loss', color='blue')
    plt.plot(eval_iterations, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_perplexity(losses):
    """Calculate perplexity from losses"""
    return [torch.exp(torch.tensor(loss)).item() for loss in losses]

def print_model_summary(model):
    """Print a summary of the model architecture"""
    print("=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"Embedding dimension: {n_embd}")
    print(f"Number of attention heads: {n_head}")
    print(f"Number of transformer layers: {n_layer}")
    print(f"Block size (context length): {block_size}")
    print(f"Dropout rate: {dropout}")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size: {model.count_parameters()/1e6:.2f}M parameters")
    print("=" * 60)

def print_training_summary(train_losses, val_losses, learning_rates):
    """Print a summary of training results"""
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best training loss: {min(train_losses):.4f}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    print(f"Final learning rate: {learning_rates[-1]:.6f}")
    
    # Calculate perplexities
    train_perplexity = torch.exp(torch.tensor(train_losses[-1])).item()
    val_perplexity = torch.exp(torch.tensor(val_losses[-1])).item()
    print(f"Final training perplexity: {train_perplexity:.2f}")
    print(f"Final validation perplexity: {val_perplexity:.2f}")
    print("=" * 60)

def save_training_history(eval_iterations, train_losses, val_losses, learning_rates, grad_norms, filename="training_history.pt"):
    """Save training history to file"""
    history = {
        'eval_iterations': eval_iterations,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'grad_norms': grad_norms,
        'config': {
            'batch_size': batch_size,
            'block_size': block_size,
            'max_iters': max_iters,
            'learning_rate': learning_rate,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout
        }
    }
    torch.save(history, filename)
    print(f"Training history saved to {filename}")

def load_training_history(filename="training_history.pt"):
    """Load training history from file"""
    history = torch.load(filename)
    return history

def analyze_text_statistics(text):
    """Analyze basic statistics of the text data"""
    print("=" * 60)
    print("TEXT STATISTICS")
    print("=" * 60)
    print(f"Total characters: {len(text):,}")
    print(f"Total words: {len(text.split()):,}")
    print(f"Unique characters: {len(set(text)):,}")
    
    # Character frequency analysis
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Sort by frequency
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost frequent characters:")
    for char, freq in sorted_chars[:10]:
        if char == ' ':
            print(f"'<space>': {freq:,} ({freq/len(text)*100:.2f}%)")
        elif char == '\n':
            print(f"'<newline>': {freq:,} ({freq/len(text)*100:.2f}%)")
        else:
            print(f"'{char}': {freq:,} ({freq/len(text)*100:.2f}%)")
    print("=" * 60)

def estimate_training_time(model, data_loader, device, num_samples=10):
    """Estimate training time per iteration"""
    model.train()
    
    times = []
    for _ in range(num_samples):
        start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        
        if device == 'cuda':
            start_time.record()
        else:
            import time
            start = time.time()
        
        # Sample batch and forward pass
        xb, yb = data_loader.get_batch('train')
        logits, loss = model(xb, yb)
        loss.backward()
        
        if device == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)  # milliseconds
        else:
            elapsed = (time.time() - start) * 1000  # convert to milliseconds
        
        times.append(elapsed)
    
    avg_time = np.mean(times)
    estimated_total = (avg_time * max_iters) / 1000 / 60  # convert to minutes
    
    print(f"Average time per iteration: {avg_time:.2f}ms")
    print(f"Estimated total training time: {estimated_total:.2f} minutes")
    
    return avg_time

def check_model_health(model):
    """Check model for common issues"""
    print("=" * 60)
    print("MODEL HEALTH CHECK")
    print("=" * 60)
    
    issues = []
    
    # Check for NaN parameters
    nan_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params += torch.isnan(param).sum().item()
            issues.append(f"NaN values in {name}")
        total_params += param.numel()
    
    if nan_params > 0:
        print(f"⚠️  Found {nan_params} NaN parameters out of {total_params}")
    else:
        print(f"✅ No NaN parameters found ({total_params:,} total parameters)")
    
    # Check parameter magnitudes
    large_params = []
    small_params = []
    for name, param in model.named_parameters():
        max_val = param.abs().max().item()
        min_val = param.abs().min().item()
        
        if max_val > 10:
            large_params.append((name, max_val))
        if min_val < 1e-6 and min_val > 0:
            small_params.append((name, min_val))
    
    if large_params:
        print(f"⚠️  Found {len(large_params)} parameters with large values (>10)")
        for name, val in large_params[:3]:  # Show first 3
            print(f"   {name}: max = {val:.2e}")
    
    if small_params:
        print(f"⚠️  Found {len(small_params)} parameters with very small values (<1e-6)")
    
    # Check gradients if available
    grad_issues = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                grad_issues += 1
                issues.append(f"NaN gradients in {name}")
    
    if grad_issues > 0:
        print(f"⚠️  Found NaN gradients in {grad_issues} parameters")
    
    if not issues:
        print("✅ Model appears healthy!")
    else:
        print(f"⚠️  Found {len(issues)} potential issues")
    
    print("=" * 60)
    
    return issues

def create_directories():
    """Create necessary directories for the project"""
    import os
    directories = [checkpoint_dir, "runs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def launch_tensorboard():
    """Launch TensorBoard (for Jupyter environments)"""
    try:
        get_ipython().run_line_magic('load_ext', 'tensorboard')
        get_ipython().run_line_magic('tensorboard', '--logdir runs')
    except NameError:
        print("TensorBoard launch is only available in Jupyter environments.")
        print("To launch TensorBoard manually, run:")
        print("tensorboard --logdir runs")