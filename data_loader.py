"""
Data loading, preprocessing, tokenization, and batching for MicroMedGPT
"""

import os
import re
import torch
from tokenizers import ByteLevelBPETokenizer
from config import *

def expand_abbreviations(text, abbr_map):
    """Expand medical abbreviations in text"""
    text = text.lower()  # lowercase first
    for abbr, full in abbr_map.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text

def normalize_units(text):
    """Normalize numbers with units for consistency"""
    # Examples: 210 mg/dL -> 210 milligrams per deciliter
    text = re.sub(r"(\d+)\s*mg/dl", r"\1 milligrams per deciliter", text)
    text = re.sub(r"(\d+)\s*mmhg", r"\1 millimeters of mercury", text)
    text = re.sub(r"(\d+)\s*kg", r"\1 kilograms", text)
    text = re.sub(r"(\d+)\s*cm", r"\1 centimeters", text)
    return text

def clean_text(text):
    """Clean text while preserving special tokens"""
    for token in special_tokens:
        text = text.replace(token, f" {token} ")
    # Remove unwanted characters
    text = re.sub(f"[^{re.escape(allowed_chars)}]", " ", text)
    # Replace multiple spaces with one
    text = re.sub("\s+", " ", text)
    return text.strip()

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    # Upload file if not present (Colab compatibility)
    if not os.path.exists(file_path):
        try:
            from google.colab import files
            uploaded = files.upload()
            for name in uploaded.keys():
                file_path = name
        except ImportError:
            raise FileNotFoundError(f"{file_path} not found and not running in Colab.")

    # Read raw text
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Apply full preprocessing pipeline
    cleaned_text = clean_text(raw_text)
    cleaned_text = expand_abbreviations(cleaned_text, abbreviation_map)
    cleaned_text = normalize_units(cleaned_text)

    print(f"Loaded, cleaned, and fully standardized dataset: {len(cleaned_text)} characters")
    return cleaned_text

def create_or_load_tokenizer(cleaned_text):
    """Create a new tokenizer or load existing one"""
    if not os.path.exists(tokenizer_vocab_file) or not os.path.exists(tokenizer_merges_file):
        print("Training new tokenizer...")
        
        # Create a temporary file with the cleaned text for training
        temp_file = "temp_training_data.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Initialize tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Train tokenizer on the temporary file
        tokenizer.train(
            files=[temp_file],
            vocab_size=tokenizer_vocab_size,
            min_frequency=tokenizer_min_frequency,
            special_tokens=special_tokens
        )
        
        # Save tokenizer
        tokenizer.save_model(".", "medical_tokenizer")
        
        # Clean up temporary file
        os.remove(temp_file)
        
        print("Tokenizer training completed.")
    else:
        print("Loading existing tokenizer...")
        # Load existing tokenizer
        tokenizer = ByteLevelBPETokenizer(tokenizer_vocab_file, tokenizer_merges_file)

    return tokenizer

def prepare_dataset(cleaned_text, tokenizer):
    """Encode text and prepare train/val splits"""
    print("Encoding text...")
    encoded_result = tokenizer.encode(cleaned_text)
    data = torch.tensor(encoded_result.ids, dtype=torch.long)

    print(f"Length of cleaned_text: {len(cleaned_text)}")
    print(f"Length of encoded data tensor: {len(data)}")

    # Get vocabulary size
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    # Verify tokenizer is working
    if vocab_size == 0 or len(data) == 0:
        raise ValueError("Tokenizer failed to initialize properly. Check your input data.")

    # Split into train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")

    return train_data, val_data, vocab_size

def adjust_hyperparameters(train_data, val_data):
    """Auto-adjust block_size and batch_size based on data size"""
    global block_size, batch_size
    
    min_split_len = min(len(train_data), len(val_data))
    if min_split_len == 0:
        raise ValueError("Training or validation data is empty after tokenization.")

    # Ensure we have enough data for the block size
    if block_size >= min_split_len:
        old_block = block_size
        block_size = max(4, min_split_len // 2)  # at least 4, but leave room for sequences
        print(f"⚠️ block_size {old_block} too large for dataset ({min_split_len} tokens). Reduced to {block_size}.")

    # Ensure we have enough data for the batch size
    max_possible_batch = min_split_len - block_size
    if batch_size > max_possible_batch:
        old_batch = batch_size
        batch_size = max(1, max_possible_batch)
        print(f"⚠️ batch_size {old_batch} too large for dataset. Reduced to {batch_size}.")

    print(f"Final block_size: {block_size}")
    print(f"Final batch_size: {batch_size}")
    
    return block_size, batch_size

def sample_batch(split, train_data, val_data):
    """Sample a batch of data for training or validation"""
    data_split = train_data if split == 'train' else val_data
    if len(data_split) <= block_size:
        raise ValueError(
            f"Split '{split}' too small: {len(data_split)} tokens for block_size={block_size}."
        )
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class DataLoader:
    """Main data loader class that encapsulates all data processing"""
    
    def __init__(self, data_file_path):
        self.data_file = data_file_path
        self.tokenizer = None
        self.train_data = None
        self.val_data = None
        self.vocab_size = None
        self.encode = None
        self.decode = None
        
    def load_and_prepare(self):
        """Complete data loading and preparation pipeline"""
        # Load and preprocess data
        cleaned_text = load_and_preprocess_data(self.data_file)
        
        # Create or load tokenizer
        self.tokenizer = create_or_load_tokenizer(cleaned_text)
        
        # Prepare dataset
        self.train_data, self.val_data, self.vocab_size = prepare_dataset(cleaned_text, self.tokenizer)
        
        # Adjust hyperparameters
        adjust_hyperparameters(self.train_data, self.val_data)
        
        # Define encode/decode functions
        self.encode = lambda s: self.tokenizer.encode(s).ids
        self.decode = lambda l: self.tokenizer.decode(l)
        
        return self.train_data, self.val_data, self.vocab_size
    
    def get_batch(self, split):
        """Get a batch of data"""
        return sample_batch(split, self.train_data, self.val_data)