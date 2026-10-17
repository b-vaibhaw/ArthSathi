import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arthasathi_model import ArthaSathiLLM, get_small_config
from model.tokenizer import ArthaSathiTokenizer

def main():
    print("Initializing a mock model checkpoint for ArthaSathi...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load tokenizer and verify
    tok_dir = "arthasathi_tokenizer"
    if not Path(tok_dir).exists():
        print("ERROR: Tokenizer directory not found. Run tokenizer training first.")
        sys.exit(1)
        
    tokenizer = ArthaSathiTokenizer(tok_dir)
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    
    # 2. Get small model config
    config = get_small_config()
    # Force match the vocab size
    config.vocab_size = len(tokenizer)
    
    # 3. Initialize model
    print("Creating the ArthaSathi LLM model...")
    model = ArthaSathiLLM(config).to(device)
    print(f"Model created. Total parameters: {model.param_count()/1e6:.1f}M")
    
    # 4. Generate mock training batch
    # Let's create some financial sentences to run a few optimization steps
    sentences = [
        "मेरा नाम राम है। मुझे कर्ज चुकाना है।",
        "My loan is 50000 rupees. How do I repay?",
        "Bajaj Finance Rs 20000 @ 26% min payment 1200",
        "kiraye ke 5000 mile. business sales recorded.",
        "GST Composition Scheme pay only 1% flat rate on sales"
    ]
    
    input_ids_list = []
    target_ids_list = []
    
    for s in sentences:
        ids = tokenizer.encode(s, add_special=True)
        # Pad or truncate to context_length (512 for small config)
        if len(ids) > config.context_length:
            ids = ids[:config.context_length]
        else:
            ids = ids + [config.pad_token_id] * (config.context_length - len(ids))
        
        # input is ids, target is ids shifted by 1
        input_ids_list.append(ids)
        target_ids_list.append(ids)
        
    x = torch.tensor(input_ids_list, dtype=torch.long).to(device)
    y = torch.tensor(target_ids_list, dtype=torch.long).to(device)
    
    # 5. Run a few mock training steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    print("Running 3 mock optimization steps...")
    for step in range(3):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}/3 | Loss: {loss.item():.4f}")
        
    # 6. Save checkpoint
    out_dir = Path("checkpoints/finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "final_ft.pt"
    
    print(f"Saving checkpoint to {out_file}...")
    model.save_checkpoint(str(out_file), step=3)
    print("Mock model checkpoint successfully saved and ready to utilize! \u2713")

if __name__ == "__main__":
    main()
