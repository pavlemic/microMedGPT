"""
Training script for MicroMedGPT
Handles training loop, logging, checkpointing, and visualization
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from config import *
from data_loader import DataLoader
from model import MicroMedGPTModel
from utils import plot_training_curves

def setup_training():
    """Setup training environment"""
    # Set random seed
    torch.manual_seed(random_seed)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(runs_dir)
    
    return writer

@torch.no_grad()
def estimate_loss(model, data_loader):
    """Estimate loss on training and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model():
    """Main training function"""
    print("Setting up training...")
    writer = setup_training()
    
    # Load and prepare data
    print("Loading data...")
    data_loader = DataLoader(data_file)
    train_data, val_data, vocab_size = data_loader.load_and_prepare()
    
    # Initialize model
    print("Initializing model...")
    model = MicroMedGPTModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    
    print(f"{model.count_parameters()/1e6:.2f}M parameters")
    
    # Training tracking
    train_losses = []
    val_losses = []
    eval_iterations = []
    learning_rates = []
    grad_norms = []
    
    print("Starting training...")
    
    # Training loop
    for iter in range(max_iters):
        # Evaluate losses every eval_interval steps
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data_loader)
            train_loss = losses['train'].item()
            val_loss = losses['val'].item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            eval_iterations.append(iter)
            print(f"Step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, iter)
            writer.add_scalar('Loss/val', val_loss, iter)

        # Training step
        xb, yb = data_loader.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Compute gradient norm before clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar("GradNorm/total", total_norm, iter)
        grad_norms.append(total_norm)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Log current learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LearningRate', current_lr, iter)
        learning_rates.append(current_lr)

    # Save final model
    print("Saving model...")
    model.save_checkpoint(model_ckpt)
    print(f"Model saved to {model_ckpt}")
    print(f"Tokenizer files saved in current directory")
    
    # Close tensorboard writer
    writer.close()
    
    # Plot training curves
    print("Generating training plots...")
    plot_training_curves(eval_iterations, train_losses, val_losses, learning_rates, grad_norms)
    
    return model, data_loader

def main():
    """Main function to run training"""
    try:
        model, data_loader = train_model()
        print("Training completed successfully!")
        return model, data_loader
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

if __name__ == "__main__":
    model, data_loader = main()