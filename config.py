"""
Configuration file for MicroMedGPT
Contains all hyperparameters, paths, and constants
"""

import os
import torch

# --------------------
# Device Configuration
# --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------
# Training Hyperparameters
# --------------------
batch_size = 16
block_size = 32  # number of tokens
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# --------------------
# Paths and Files
# --------------------
checkpoint_dir = "checkpoints"
runs_dir = "runs/microMedGPT"
data_file = 'microMedGPT.txt'
model_ckpt = os.path.join(checkpoint_dir, "microMedGPT_model.pt")
tokenizer_file = "medical_tokenizer.json"
tokenizer_vocab_file = "medical_tokenizer-vocab.json"
tokenizer_merges_file = "medical_tokenizer-merges.txt"

# --------------------
# Tokenizer Configuration
# --------------------
tokenizer_vocab_size = 5000
tokenizer_min_frequency = 2
special_tokens = ["<PAD>", "<UNK>", "<SEP>", "<note>", "<report>"]

# --------------------
# Text Processing Configuration
# --------------------
allowed_chars = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,:;!?()/- \n%"
)

abbreviation_map = {
    "fbg": "fasting blood glucose",
    "hba1c": "hemoglobin a1c",
    "pt": "patient",
    "wt": "weight",
    "rul": "right upper lobe",
    "lul": "left upper lobe",
    "ll": "left lower lobe",
    "rl": "right lower lobe",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "c/o": "complains of",
    "hx": "history",
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "copd": "chronic obstructive pulmonary disease",
    "sob": "shortness of breath",
    "npo": "nothing by mouth",
    "po": "by mouth",
    "iv": "intravenous",
    "q": "every",
    "tid": "three times a day",
    "bid": "twice a day",
    "prn": "as needed",
    "stat": "immediately",
    "o2": "oxygen",
    # Lab abbreviations / units
    "mg/dl": "milligrams per deciliter",
    "mmhg": "millimeters of mercury",
    "bpm": "beats per minute",
    "l": "liter",
    "kg": "kilogram",
    "cm": "centimeter",
}

# --------------------
# Generation Configuration
# --------------------
default_max_tokens = 200
default_temperature = 0.8
default_top_k = 50
default_top_p = 0.9
default_stop_tokens = ["<report>", "---"]

# --------------------
# Random Seed
# --------------------
random_seed = 167