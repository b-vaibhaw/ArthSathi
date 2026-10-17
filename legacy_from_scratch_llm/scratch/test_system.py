import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_tests():
    print("=" * 60)
    print("ArthaSathi AI Integration Test Suite")
    print("=" * 60)
    
    # 1. Verify dependencies
    try:
        from fastapi.testclient import TestClient
        print("FastAPI TestClient found.")
    except ImportError:
        print("ERROR: fastapi testclient not available. Run: pip install httpx")
        sys.exit(1)
        
    # 2. Import app and create client
    from api.main import app, state, cfg
    
    # Force config to use cpu for testing
    cfg.DEVICE = "cpu"
    cfg.MODEL_PATH = "checkpoints/finetune/final_ft.pt"
    cfg.TOKENIZER_DIR = "arthasathi_tokenizer"
    
    # Delete old database file for idempotency
    db_file = Path(cfg.DB_PATH)
    if db_file.exists():
        try:
            db_file.unlink()
            print(f"Cleared existing database file: {db_file}")
        except Exception as e:
            print(f"Could not clear database file: {e}")
            
    print("\nStarting TestClient...")
    # Using with-block triggers lifespan startup & shutdown handlers (loading models)
    with TestClient(app) as client:
        print("\n--- Endpoint: GET /health ---")
        res = client.get("/health")
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"
        assert res.json()["model"] is True
        assert res.json()["tokenizer"] is True
        
        print("\n--- Endpoint: POST /calculate/emi ---")
        res = client.post("/calculate/emi?principal=100000&annual_rate=18&tenure_months=24")
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["emi"] == 4992.41
        assert res.json()["total_interest"] == 19817.84
        
        print("\n--- Endpoint: POST /calculate/tax ---")
        res = client.post("/calculate/tax?annual_income=800000&regime=new")
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["total_tax"] == 36400.0
        assert res.json()["effective_rate"] == 4.5
        
        print("\n--- Endpoint: POST /calculate/gst ---")
        res = client.post("/calculate/gst?annual_turnover=3500000&supplies_type=goods")
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["status"] == "OPTIONAL_COMPOSITION"
        assert res.json()["gst_rate"] == 1.0
        
        print("\n--- Endpoint: POST /user/income ---")
        res = client.post("/user/income", json={"user_id": "test_user_ci", "monthly_income": 30000.0})
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["status"] == "updated"
        
        print("\n--- Endpoint: POST /user/debt ---")
        res = client.post("/user/debt", json={
            "user_id": "test_user_ci",
            "name": "Moneylender Shyam",
            "principal": 15000.0,
            "annual_rate": 36.0,
            "min_payment": 600.0,
            "lender_type": "moneylender"
        })
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert res.json()["status"] == "added"
        
        print("\n--- Endpoint: GET /user/{user_id}/profile ---")
        res = client.get("/user/test_user_ci/profile")
        print(f"Status: {res.status_code}")
        profile = res.json()
        print(f"Response (summary): Income={profile['monthly_income']}, Total Debt={profile['total_debt']}, DTI={profile['debt_to_income']}%")
        assert res.status_code == 200
        assert profile["monthly_income"] == 30000.0
        assert profile["total_debt"] == 15000.0
        assert profile["debt_to_income"] == 2.0
        
        print("\n--- Endpoint: POST /chat (LLM Inference + RAG) ---")
        res = client.post("/chat", json={
            "user_id": "test_user_ci",
            "message": "bhai mera moneylender shyam ka 15000 ka loan hai, kya karna chahiye?",
            "language": "hi"
        })
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200
        assert "response" in res.json()
        assert len(res.json()["response"]) > 10
        
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY! (Mock LLM utilized)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
