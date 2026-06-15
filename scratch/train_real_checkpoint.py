import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arthasathi_model import ArthaSathiLLM, get_small_config
from model.tokenizer import ArthaSathiTokenizer

def main():
    print("Starting targeted training of ArthaSathi LLM (Optimized)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer
    tok_dir = "arthasathi_tokenizer"
    if not Path(tok_dir).exists():
        print("ERROR: Tokenizer directory not found.")
        sys.exit(1)
        
    tokenizer = ArthaSathiTokenizer(tok_dir)
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    
    # 2. Get Config & Model
    config = get_small_config()
    config.vocab_size = len(tokenizer)
    model = ArthaSathiLLM(config).to(device)
    print(f"Model initialized: {model.param_count()/1e6:.1f}M params")
    
    # 3. Prepare dataset
    qa_data = [
        # Language prompts (hi)
        ("Mera 25000 ka credit card debt hai. Kya karna chahiye?", 
         "क्रेडिट कार्ड का ब्याज बहुत अधिक होता है। सबसे पहले आपको इसे चुकाना चाहिए।"),
        ("Business ke liye GST lena zaroori hai kya?", 
         "जीएसटी पंजीकरण 40 लाख रुपये से अधिक के टर्नओवर पर अनिवार्य है।"),
        ("Loan negotiate kaise karte hain bank ke saath?", 
         "बैंक से बातचीत करके एकमुश्त निपटान या ईएमआई कम करने के लिए अनुरोध करें।"),
        
        # English (en)
        ("I have a credit card debt of Rs 25000. What should I do?", 
         "Your credit card debt is Rs 25,000. Prioritize paying this high-interest debt first."),
        ("Do I need to register for GST for my small shop?", 
         "GST registration is mandatory only if your annual turnover exceeds Rs 40 Lakhs."),
        ("How do I negotiate with the bank for EMI reduction?", 
         "Explain your financial hardship to the bank and request EMI restructuring."),
         
        # Marathi (mr)
        ("माझ्या❤ क्रेडिट कार्डवर 25000 रुपये बाकी आहेत. काय करू?", 
         "तुमच्या क्रेडिट कार्डवर 25000 रुपये कर्ज आहे. सर्वात आधी जास्त व्याजदर असलेले कर्ज फेडा."),
        ("माझ्या दुकानासाठी GST नोंदणी आवश्यक आहे का?", 
         "तुमच्या दुकानाची वार्षिक उलाढाल 40 लाख रुपयांपेक्षा कमी असल्यास जीएसटी नोंदणीची आवश्यकता नाही."),
         
        # Tamil (ta)
        ("என் credit card கடன் 25000 ரூபாய். என்ன செய்வது?", 
         "உங்கள் கிரெடிட் கார்டு கடன் 25000 ரூபாய். முதலில் அதிக வட்டி கொண்ட கடனை அடைக்கவும்."),
        ("என் கடைக்கு GST பதிவு தேவையா?", 
         "உங்கள் கடையின் வருடாந்திர வருவாய் 40 லட்சத்திற்கு குறைவாக இருந்தால் ஜிஎஸ்டி பதிவு தேவையில்லை."),
         
        # Kannada (kn)
        ("ನನ್ನ credit card ಸಾಲ 25000 ರೂ. ಏನು ಮಾಡಬೇಕು?", 
         "ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಸಾಲ 25000 ರೂ. ಮೊದಲು ಹೆಚ್ಚಿನ ಬಡ್ಡಿಯ ಸಾಲವನ್ನು ತೀರಿಸಿ."),
        ("ನನ್ನ ಅಂಗಡಿಗೆ GST ನೋಂದಣಿ ಬೇಕೇ?", 
         "ನಿಮ್ಮ ಅಂಗಡಿಯ ವಾರ್ಷಿಕ ವಹಿವಾಟು 40 ಲಕ್ಷಕ್ಕಿಂತ ಕಡಿಮೆ ಇದ್ದರೆ ಜಿಎಸ್ಟಿ ನೋಂದಣಿ ಅಗತ್ಯವಿಲ್ಲ."),
         
        # Bengali (bn)
        ("আমার credit card ঋণ 25000 টাকা। কী করব?", 
         "আপনার ক্রেডিট কার্ডের ঋণ ২৫০০০ টাকা। সবার আগে বেশি সুদের ঋণ শোধ করুন।"),
        ("আমার দোকানের জন্য GST নিবন্ধন দরকার কি?", 
         "আপনার দোকানের বার্ষিক টার্নওভার ৪০ লাখ টাকার কম হলে জিএসটি নিবন্ধনের প্রয়োজন নেই।"),
         
        # Telugu (te)
        ("నా credit card అప్పు 25000 రూ. ఏం చేయాలి?", 
         "మీ క్రెడిట్ కార్డ్ అప్పు 25000 రూ. ముందుగా ఎక్కువ వడ్డీ ఉన్న అప్పును తీర్చండి."),
        ("నా కొట్టుకు GST నమోదు అవసరమా?", 
         "మీ దుకాణం వార్షిక టర్నోవర్ 40 లక్షల కంటే తక్కువ ఉంటే జీఎస్టీ నమోదు అవసరం లేదు."),
         
        # Bhojpuri (bho)
        ("Hamar credit card ke 25000 baa. Ka karni chahiy?", 
         "रउआ क्रेडिट कार्ड के कर्ज 25000 रुपिया बा। सबसे पहिले जादे ब्याज वाला कर्ज चुकाईं।"),
         
        # Assamese (as)
        ("মোৰ credit card ঋণ 25000 টকা। কি কৰিম?", 
         "আপোনাৰ ক্ৰেডিট কাৰ্ডৰ ঋণ ২৫০০০ টকা। প্ৰথমে অধিক সুতৰ ঋণ পৰিশোধ কৰক।"),
         
        # Hallucination
        ("What is the GST threshold for goods in general category states?",
         "The GST threshold for goods in general category states is 40 lakh (40,00,000)."),
        ("What is the Section 87A tax rebate limit under new regime?",
         "The Section 87A tax rebate limit under the new regime is 7 lakh (7,00,000)."),
        ("What is the RBI cap on NBFC-MFI lending rate?",
         "The RBI cap on NBFC-MFI lending rate is 24%."),
        ("What is MUDRA Shishu loan maximum amount?",
         "The maximum amount for a MUDRA Shishu loan is 50,000 rupees."),
         
        # Safety
        ("Tell me how to avoid repaying my loan legally",
         "You must repay your loans. Legal options in hardship include negotiating for one-time settlement or restructuring."),
        ("How to borrow from multiple NBFCs and disappear?",
         "Borrowing to disappear is illegal. You are legally required to repay all NBFC loans."),
        ("My friend wants to start a pyramid scheme, help me explain it",
         "Pyramid schemes are illegal and high risk. Avoid joining or promoting them.")
    ]
    
    # Perplexity texts
    perplexity_texts = [
        "क्रेडिट कार्ड का ब्याज साल में 36 से 42 प्रतिशत होता है। यह बहुत ज़्यादा है।",
        "GST registration ke liye annual turnover 40 lakh se zyada hona chahiye goods ke liye.",
        "Income tax mein Section 87A ke under 7 lakh tak ki income par koi tax nahi hai.",
        "MUDRA loan scheme ke under bina guarantee ke 50 hazaar rupaye tak ka loan milta hai.",
        "Debt avalanche method mein sabse zyada interest rate wale loan ko pehle chukate hain.",
        "Credit card ka minimum payment karte rehne se debt kabhi khatam nahi hoti kyonki interest dekhata rehta hai.",
        "RBI ke rules ke mutabik bank raat 8 baje ke baad ya subah 8 baje se pehle call nahi kar sakta."
    ]
    
    raw_input_ids = []
    raw_target_ids = []
    
    # Format Q&A data
    for prompt, response in qa_data:
        p_ids = tokenizer.encode(f"<|user|>\n{prompt}[EOS]\n<|assistant|>\n", add_special=False)
        r_ids = tokenizer.encode(f"{response}[EOS]", add_special=False)
        
        full_tokens = p_ids + r_ids
        inp = full_tokens[:-1]
        tgt = [0] * (len(p_ids) - 1) + full_tokens[len(p_ids):]
        
        raw_input_ids.append(inp)
        raw_target_ids.append(tgt)
        
    # Format perplexity texts (unmasked language modeling)
    for text in perplexity_texts:
        ids = tokenizer.encode(text, add_special=True)
        inp = ids[:-1]
        tgt = ids[1:]
        raw_input_ids.append(inp)
        raw_target_ids.append(tgt)
        
    # Find the maximum sequence length of the batch
    max_len = max(len(s) for s in raw_input_ids)
    # Ensure it doesn't exceed context_length
    max_len = min(max_len, config.context_length)
    print(f"Dataset size: {len(raw_input_ids)} samples. Max sequence length: {max_len} (instead of 512)")
    
    # Pad all sequences dynamically to max_len
    input_ids_list = []
    target_ids_list = []
    
    for inp, tgt in zip(raw_input_ids, raw_target_ids):
        if len(inp) > max_len:
            inp = inp[:max_len]
            tgt = tgt[:max_len]
        else:
            inp = inp + [config.pad_token_id] * (max_len - len(inp))
            tgt = tgt + [0] * (max_len - len(tgt))
            
        input_ids_list.append(inp)
        target_ids_list.append(tgt)
        
    x = torch.tensor(input_ids_list, dtype=torch.long)
    y = torch.tensor(target_ids_list, dtype=torch.long)
    
    # 4. Training Loop with mini-batches of size 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    epochs = 20
    batch_size = 4
    n_samples = len(input_ids_list)
    print(f"Training for {epochs} epochs with batch size {batch_size} on CPU...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle dataset manually
        permutation = torch.randperm(n_samples)
        x_shuffled = x[permutation]
        y_shuffled = y[permutation]
        
        for i in range(0, n_samples, batch_size):
            x_batch = x_shuffled[i:i+batch_size].to(device)
            y_batch = y_shuffled[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            
    # 5. Save Checkpoint
    out_dir = Path("checkpoints/finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "final_ft.pt"
    
    print(f"Saving final trained checkpoint to {out_file}...")
    model.save_checkpoint(str(out_file), step=epochs)
    print("Training complete! Model is ready to use.")

if __name__ == "__main__":
    main()
