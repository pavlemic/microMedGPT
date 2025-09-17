# MicroMedGPT

A small-scale GPT model designed for medical report generation from clinical notes.

## Project Structure

```
microMedGPT/
│── config.py          # Configuration, hyperparameters, and paths
│── data_loader.py     # Data preprocessing, tokenization, and batching
│── model.py           # GPT model architecture and components
│── train.py           # Training loop, logging, and checkpointing
│── generate.py        # Inference and report generation
│── utils.py           # Utility functions for plotting and analysis
│── requirements.txt   # Python dependencies
│── checkpoints/       # Directory for saved models
│── runs/              # TensorBoard logs
│── microMedGPT.txt    # Training dataset (not included)
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd microMedGPT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir checkpoints runs
```

## Usage

### Training

To train the model, place your dataset in `microMedGPT.txt` and run:

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train or load a BPE tokenizer
- Train the GPT model
- Save checkpoints and generate training plots
- Log metrics to TensorBoard

### Generation

After training, you can generate reports in several ways:

**Interactive generation:**
```bash
python generate.py interactive
```

**Batch generation example:**
```bash
python generate.py batch
```

**Programmatic usage:**
```python
from generate import generate_single_report

report = generate_single_report("F 45 yo, polyuria, polydipsia. FBG 210 mg/dL <SEP>")
print(report)
```

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir runs
```

## Configuration

Edit `config.py` to modify:
- Model architecture (layers, heads, embedding size)
- Training hyperparameters (learning rate, batch size, iterations)
- File paths and tokenizer settings
- Generation parameters

## Model Architecture

- **Type:** Transformer-based language model (GPT-style)
- **Size:** ~0.2M parameters (configurable)
- **Context:** 32 tokens (configurable)
- **Vocabulary:** 5000 BPE tokens
- **Layers:** 4 transformer blocks
- **Heads:** 4 attention heads
- **Embedding:** 64 dimensions

## Features

- **Medical text preprocessing:** Expands medical abbreviations and normalizes units
- **Custom tokenization:** BPE tokenizer trained on medical text
- **Advanced sampling:** Supports temperature, top-k, and top-p sampling
- **Comprehensive logging:** TensorBoard integration with loss, gradients, and learning rate tracking
- **Model checkpointing:** Automatic saving and loading of trained models
- **Interactive generation:** Command-line interface for report generation
- **Training visualization:** Automatic plotting of training curves
- **Health checks:** Model parameter and gradient monitoring

## Data Format

The model expects clinical notes followed by a separator token `<SEP>` and then the corresponding medical report. Example:

```
F 45 yo, polyuria, polydipsia. FBG 210 mg/dL, HbA1c 8.5%. <SEP> 
ASSESSMENT AND PLAN: 45-year-old female presents with classic symptoms of diabetes mellitus...

<note> M 62 yo, chest pain, SOB <SEP> 
CLINICAL IMPRESSION: Acute coronary syndrome...
```

## Troubleshooting

**Empty tokenization error:**
- Ensure your dataset file exists and contains text
- Check that the text preprocessing doesn't remove all content
- Verify special tokens are properly formatted

**Memory issues:**
- Reduce `batch_size` or `block_size` in `config.py`
- Use a smaller model (reduce `n_embd`, `n_head`, or `n_layer`)

**Poor generation quality:**
- Train for more iterations
- Increase model size
- Adjust generation parameters (temperature, top_k, top_p)
- Ensure training data is sufficient and properly formatted

