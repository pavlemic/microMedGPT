"""
Inference and report generation for MicroMedGPT
"""

import torch
from config import *
from data_loader import DataLoader
from model import MicroMedGPTModel

def generate_reports(model, encode_fn, decode_fn, notes, max_tokens=None, n=3, temperature=None, top_k=None, top_p=None, stop_tokens=None):
    """
    Generate multiple medical reports from clinical notes
    
    Args:
        model: trained MicroMedGPT model
        encode_fn: function to encode text to tokens
        decode_fn: function to decode tokens to text
        notes: input clinical notes string
        max_tokens: maximum tokens to generate
        n: number of reports to generate
        temperature: sampling temperature
        top_k: top-k sampling parameter
        top_p: top-p sampling parameter
        stop_tokens: list of tokens to stop generation
        
    Returns:
        list of generated report strings
    """
    # Use defaults if not provided
    max_tokens = max_tokens or default_max_tokens
    temperature = temperature or default_temperature
    top_k = top_k or default_top_k
    top_p = top_p or default_top_p
    stop_tokens = stop_tokens or default_stop_tokens
    
    # Encode input notes
    context = torch.tensor([encode_fn(notes)], dtype=torch.long, device=device)
    outputs = []

    print(f"Generating {n} reports with the following parameters:")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"  Stop tokens: {stop_tokens}")
    
    model.eval()
    with torch.no_grad():
        for i in range(n):
            print(f"Generating report {i+1}/{n}...")
            gen_text = model.generate(
                context,
                max_new_tokens=max_tokens,
                decode_fn=decode_fn,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Apply stop tokens
            for stop_token in stop_tokens:
                if stop_token in gen_text:
                    gen_text = gen_text.split(stop_token)[0].strip()
            
            outputs.append(gen_text)

    return outputs

def load_trained_model(model_path, data_loader):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: path to model checkpoint
        data_loader: DataLoader instance with tokenizer
        
    Returns:
        loaded model
    """
    model = MicroMedGPTModel(data_loader.vocab_size).to(device)
    model.load_checkpoint(model_path)
    model.eval()
    return model

def interactive_generation():
    """Interactive report generation interface"""
    print("Loading model and tokenizer...")
    
    # Load data loader (for tokenizer)
    data_loader = DataLoader(data_file)
    data_loader.load_and_prepare()
    
    # Load trained model
    model = load_trained_model(model_ckpt, data_loader)
    
    print("Model loaded successfully!")
    print("Enter clinical notes followed by '<SEP>' to generate reports.")
    print("Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_input = input("Enter clinical notes: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        if not user_input:
            print("Please enter some clinical notes.")
            continue
            
        # Add separator if not present
        if "<SEP>" not in user_input:
            user_input += " <SEP>"
        
        try:
            # Generate reports
            reports = generate_reports(
                model,
                data_loader.encode,
                data_loader.decode,
                user_input,
                n=3
            )
            
            # Display results
            print(f"\n{'='*60}")
            print("GENERATED MEDICAL REPORTS:")
            print(f"{'='*60}")
            
            for i, report in enumerate(reports, 1):
                print(f"\nReport {i}:")
                print("-" * 40)
                print(report)
            
            print(f"\n{'='*60}\n")
            
        except Exception as e:
            print(f"Error generating reports: {e}")
            print("Please try again with different input.\n")

def batch_generation_example():
    """Example function showing batch report generation"""
    print("Loading model and tokenizer for batch generation...")
    
    # Load data loader (for tokenizer)
    data_loader = DataLoader(data_file)
    data_loader.load_and_prepare()
    
    # Load trained model
    model = load_trained_model(model_ckpt, data_loader)
    
    # Example clinical notes
    example_notes = [
        "F 45 yo, polyuria, polydipsia. FBG 210 mg/dL, HbA1c 8.5%. <SEP>",
        "M 62 yo, chest pain, SOB. BP 160/90, HR 95. ECG shows ST elevation. <SEP>",
        "F 28 yo, fever, cough, fatigue. Temperature 101.2F, O2 sat 96%. <SEP>"
    ]
    
    print("Generating reports for example cases...\n")
    
    for i, notes in enumerate(example_notes, 1):
        print(f"Case {i}: {notes.replace('<SEP>', '').strip()}")
        print("=" * 50)
        
        try:
            reports = generate_reports(
                model,
                data_loader.encode,
                data_loader.decode,
                notes,
                n=2,  # Generate 2 reports per case
                max_tokens=150
            )
            
            for j, report in enumerate(reports, 1):
                print(f"\nGenerated Report {j}:")
                print("-" * 30)
                print(report)
                
        except Exception as e:
            print(f"Error generating reports for case {i}: {e}")
            
        print("\n" + "="*70 + "\n")

def generate_single_report(notes, **kwargs):
    """
    Generate a single report from clinical notes
    
    Args:
        notes: clinical notes string
        **kwargs: generation parameters
        
    Returns:
        single generated report string
    """
    # Load data loader and model
    data_loader = DataLoader(data_file)
    data_loader.load_and_prepare()
    model = load_trained_model(model_ckpt, data_loader)
    
    # Generate single report
    reports = generate_reports(
        model,
        data_loader.encode,
        data_loader.decode,
        notes,
        n=1,
        **kwargs
    )
    
    return reports[0] if reports else ""

def main():
    """Main function for generation"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_generation()
        elif sys.argv[1] == "batch":
            batch_generation_example()
        else:
            print("Usage: python generate.py [interactive|batch]")
    else:
        print("MicroMedGPT Report Generation")
        print("Usage:")
        print("  python generate.py interactive  - Interactive generation")
        print("  python generate.py batch       - Batch generation example")
        print("\nOr import functions for custom usage:")
        print("  from generate import generate_reports, generate_single_report")

if __name__ == "__main__":
    main()