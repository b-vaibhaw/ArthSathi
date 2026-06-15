import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app, state, cfg, process_message
from fastapi.testclient import TestClient

def test_flexibility():
    print("=" * 70)
    print("Testing ArthaSathi Chatbot Flexibility & Multilingual Reasoning")
    print("=" * 70)
    
    # Configure test environment
    cfg.DEVICE = "cpu"
    cfg.MODEL_PATH = "checkpoints/finetune/final_ft.pt"
    cfg.TOKENIZER_DIR = "arthasathi_tokenizer"
    cfg.DB_PATH = "test_chat_users.db"
    
    # Clean old test DB
    db_file = Path(cfg.DB_PATH)
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass
            
    # Test queries across different languages and modules
    test_cases = [
        {
            "lang": "Hindi (Code-switched)",
            "user_id": "user_hindi_1",
            "message": "bhai mera credit card ka 40000 ka loan chal raha hai aur meri salary 15000 hai. Kuch tarika batao please."
        },
        {
            "lang": "English",
            "user_id": "user_english_1",
            "message": "The bank agents are calling me at 10 PM in the night. Is this allowed? What are my legal rights?"
        },
        {
            "lang": "Marathi",
            "user_id": "user_marathi_1",
            "message": "माझा भाजीचा गाडा आहे, रोज ३०० रुपये माल खरेदी करतो आणि ५० रुपये खर्च होतो. मला भाजी किती रुपयात विकावी लागेल?"
        },
        {
            "lang": "Hindi (GST Query)",
            "user_id": "user_hindi_2",
            "message": "Mera annual turnover 25 Lakh hai, kya mujhe GST register karna padega?"
        }
    ]
    
    # Start app context to load models
    with TestClient(app) as client:
        print("\nFastAPI app loaded. Running test queries...\n")
        
        for idx, tc in enumerate(test_cases, 1):
            try:
                print(f"\n[{idx}] Testing Language: {tc['lang']}")
                print(f"User Query : \"{tc['message']}\"")
            except UnicodeEncodeError:
                safe_msg = tc['message'].encode('ascii', 'backslashreplace').decode('ascii')
                print(f"\n[{idx}] Testing Language: {tc['lang']}")
                print(f"User Query : \"{safe_msg}\"")
            
            response = client.post("/chat", json={
                "user_id": tc["user_id"],
                "message": tc["message"]
            })
            
            if response.status_code == 200:
                answer = response.json()["response"]
                print("-" * 50)
                try:
                    print(f"ArthaSathi Assistant Reply:\n{answer}")
                except UnicodeEncodeError:
                    safe_ans = answer.encode('ascii', 'backslashreplace').decode('ascii')
                    print(f"ArthaSathi Assistant Reply:\n{safe_ans}")
                print("-" * 50)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
    # Clean up test DB
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    test_flexibility()
