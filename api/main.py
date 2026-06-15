"""
ArthaSathi — FastAPI Application Server
Handles: WhatsApp webhook, SMS fallback, REST API
Integrates: LLM inference + RAG + Financial Engines + Voice Pipeline
Deploy: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os, sys, json, asyncio, hashlib, hmac, time
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arthasathi_model import ArthaSathiLLM
from rag.rag_engine import ArthaSathiRAG
from engines.financial_engines import ArthaSathiAdvisor, Debt, UserFinancialProfile
from voice.voice_pipeline import VoiceHandler


# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────

class Config:
    MODEL_PATH      = os.getenv("MODEL_PATH",      "checkpoints/finetune/final_ft.pt")
    TOKENIZER_DIR   = os.getenv("TOKENIZER_DIR",   "arthasathi_tokenizer")
    RAG_INDEX_PATH  = os.getenv("RAG_INDEX_PATH",  "rag_index")
    DB_PATH         = os.getenv("DB_PATH",          "arthasathi_users.db")
    DEVICE          = os.getenv("DEVICE",           "cuda")

    # WhatsApp (Meta Cloud API)
    WA_PHONE_ID     = os.getenv("WA_PHONE_ID",     "")
    WA_ACCESS_TOKEN = os.getenv("WA_ACCESS_TOKEN", "")
    WA_VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN", "arthasathi_secret_2024")
    WA_API_URL      = "https://graph.facebook.com/v18.0/{phone_id}/messages"

    # SMS (Twilio or MSG91)
    SMS_PROVIDER    = os.getenv("SMS_PROVIDER",    "twilio")
    TWILIO_SID      = os.getenv("TWILIO_SID",      "")
    TWILIO_TOKEN    = os.getenv("TWILIO_TOKEN",     "")
    TWILIO_NUMBER   = os.getenv("TWILIO_NUMBER",    "")
    MSG91_API_KEY   = os.getenv("MSG91_API_KEY",    "")
    MSG91_SENDER_ID = os.getenv("MSG91_SENDER_ID",  "ARTHAS")

    # Inference
    MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "300"))
    TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.75"))
    TOP_K           = int(os.getenv("TOP_K", "50"))
    TOP_P           = float(os.getenv("TOP_P", "0.9"))


cfg = Config()


# ─────────────────────────────────────────────────────────────
# 2. GLOBAL STATE
# ─────────────────────────────────────────────────────────────

class AppState:
    model:    Optional[ArthaSathiLLM]    = None
    tokenizer = None
    rag:      Optional[ArthaSathiRAG]    = None
    advisor:  Optional[ArthaSathiAdvisor] = None
    voice:    Optional[VoiceHandler]     = None
    ready:    bool = False


state = AppState()


# ─────────────────────────────────────────────────────────────
# 3. APP STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and services on startup"""
    print("[ArthaSathi] Starting up...")
    
    # Resolve device fallback based on PyTorch CUDA availability
    import torch
    if cfg.DEVICE == "cuda" and not torch.cuda.is_available():
        print("[ArthaSathi] CUDA requested but not available. Falling back to 'cpu'.")
        cfg.DEVICE = "cpu"
    else:
        print(f"[ArthaSathi] Using device: {cfg.DEVICE}")

    # Load tokenizer
    try:
        from tokenizers import Tokenizer
        tok_path = Path(cfg.TOKENIZER_DIR) / "tokenizer.json"
        if tok_path.exists():
            state.tokenizer = Tokenizer.from_file(str(tok_path))
            print(f"[ArthaSathi] Tokenizer loaded: {state.tokenizer.get_vocab_size()} vocab")
        else:
            print(f"[ArthaSathi] WARNING: Tokenizer not found at {tok_path}")
    except Exception as e:
        print(f"[ArthaSathi] Tokenizer load failed: {e}")

    # Load LLM
    try:
        if Path(cfg.MODEL_PATH).exists():
            state.model = ArthaSathiLLM.from_checkpoint(cfg.MODEL_PATH, device=cfg.DEVICE)
            state.model.eval()
            print(f"[ArthaSathi] LLM loaded: {state.model.param_count()/1e6:.1f}M params")
        else:
            print(f"[ArthaSathi] WARNING: Model not found at {cfg.MODEL_PATH}")
            print("[ArthaSathi] Run training first. API will return template responses for now.")
    except Exception as e:
        print(f"[ArthaSathi] LLM load failed: {e}")

    # Load RAG
    try:
        state.rag = ArthaSathiRAG(
            index_path=cfg.RAG_INDEX_PATH,
            db_path=cfg.DB_PATH
        )
        print("[ArthaSathi] RAG engine ready")
        
        # Initialize auth tables
        if state.rag and state.rag.memory:
            db = state.rag.memory.db
            db.execute("""
                CREATE TABLE IF NOT EXISTS auth_users (
                    username      TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    user_id       TEXT NOT NULL
                )
            """)
            db.commit()
            print("[ArthaSathi] Auth database tables prepared")
    except Exception as e:
        print(f"[ArthaSathi] RAG load failed: {e}")

    # Load financial advisor
    state.advisor = ArthaSathiAdvisor()
    print("[ArthaSathi] Financial advisor ready")

    # Load voice handler (optional — only if Whisper installed)
    try:
        state.voice = VoiceHandler(whisper_size="small", tts_engine="gtts")
        print("[ArthaSathi] Voice handler ready")
    except Exception as e:
        print(f"[ArthaSathi] Voice handler not available: {e}")

    state.ready = True
    print("[ArthaSathi] All systems ready!")

    yield  # App runs here

    print("[ArthaSathi] Shutting down...")


# ─────────────────────────────────────────────────────────────
# 4. FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ArthaSathi AI",
    description="Micro-entrepreneur financial coach + debt advisor for 700M Indians",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def get_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return PlainTextResponse("ArthaSathi API Server is running. Static files not found.")


# ─────────────────────────────────────────────────────────────
# 5. CORE LLM INFERENCE
# ─────────────────────────────────────────────────────────────

def _generate_smart_response(user_message: str, user_id: str, language: str,
                             intent: Optional[str], engine_result: Optional[Dict],
                             rag_docs: List[Dict]) -> Optional[str]:
    """
    Generate accurate structured financial answers in English, Hindi, or Marathi
    based on engine calculations and RAG facts.
    """
    msg_lower = user_message.lower()

    # A. TRANSACTION INTENT
    if intent == "transaction" and engine_result and "parsed_transaction" in engine_result:
        txn = engine_result["parsed_transaction"]
        if txn:
            t_type = txn.get("type", "expense")
            amt = txn.get("amount", 0)
            cat = txn.get("category", "other")
            desc = txn.get("description", "")
            
            if language == "hi":
                return (f"Main aapka transaction record kar liya hai! \n\n"
                        f"\u2022 **Type**: {t_type.upper()}\n"
                        f"\u2022 **Amount**: Rs {amt:,}\n"
                        f"\u2022 **Category**: {cat}\n"
                        f"\u2022 **Detail**: {desc}\n\n"
                        f"Aapki 'Profile & Memory' tab automatic update ho chuki hai.")
            elif language == "mr":
                # Marathi translation
                return (f"\u0924\u094c\u092e\u091a\u093e \u0935\u094d\u092f\u0935\u0939\u093e\u0930 \u092f\u0936\u0938\u094d\u0935\u0940\u0930\u093f\u0924\u094d\u092f\u093e \u0928\u094b\u0902\u0926\u0935\u0932\u093e \u0917\u094d\u092f\u093e\u0932\u093e \u0906\u0939\u094d\u092f\u093e! \n\n"
                        f"\u2022 **\u092a\u094d\u0930\u0915\u093e\u0930**: {t_type.upper()}\n"
                        f"\u2022 **\u0930\u0915\u094d\u0915\u092e**: Rs {amt:,}\n"
                        f"\u2022 **\u0935\u0930\u094d\u0917**: {cat}\n"
                        f"\u2022 **\u0924\u092a\u0936\u0940\u0932**: {desc}\n\n"
                        f"\u0924\u094c\u092e\u091a\u0947 \u092a\u094d\u0930\u094b\u092b\u093e\u0908\u0932 \u0930\u094d\u0947\u0915\u094d\u094a\u0930\u094d\u0921 \u0905\u092a\u0921\u094d\u0947\u091f \u091c\u093e\u0932\u094d\u0947 \u0906\u0939\u094d\u092f\u0947.")
            else:
                return (f"Transaction recorded successfully!\n\n"
                        f"\u2022 **Type**: {t_type.upper()}\n"
                        f"\u2022 **Amount**: Rs {amt:,}\n"
                        f"\u2022 **Category**: {cat}\n"
                        f"\u2022 **Description**: {desc}\n\n"
                        f"It has been synced with your database profile in real-time.")

    # B. PRICING QUERY INTENT
    if intent == "pricing_query":
        import re
        nums = re.findall(r'\b\d+\b', user_message)
        if nums:
            cog = float(nums[0])
            opex = float(nums[1]) if len(nums) > 1 else 50.0
            margin = float(nums[2]) if len(nums) > 2 else 20.0
            
            total_cost = cog + opex
            suggested = total_cost * 1.22
            target = total_cost * (1 + margin / 100)
            breakeven = total_cost * 1.05
            
            if language == "hi":
                return (f"Aapke business ke liye suggested price analysis:\n\n"
                        f"\u2022 **Product Cost (Goods)**: Rs {cog:,}\n"
                        f"\u2022 **Operating Expense**: Rs {opex:,}\n"
                        f"\u2022 **Total Production Cost**: Rs {total_cost:,}\n\n"
                        f"**Salah**:\n"
                        f"\u2022 **Suggested Selling Price**: Rs {suggested:.0f} (Ispe lagbhag 22% margin milega).\n"
                        f"\u2022 **Target Price (at {margin:.0f}% margin)**: Rs {target:.0f}\n"
                        f"\u2022 **Breakeven (Kam se kam daam)**: Rs {breakeven:.0f} (Isse kam pe mat bechein, nuksaan hoga).\n\n"
                        f"**Tip**: Grahak ko bundeling deals dekar bikri badhayein.")
            elif language == "mr":
                # Marathi translation
                return (f"\u0924\u094c\u092e\u091a\u094d\u092f\u093e \u0935\u094d\u092f\u0935\u0938\u093e\u092f\u093e\u0938\u093e\u0920\u0940 \u0915\u093f\u0902\u092e\u0924 \u0928\u093f\u0936\u094d\u091a\u093f\u0924\u0940\u091a\u0947 \u0935\u093f\u0936\u094d\u0932\u094d\u0947\u0937\u0923:\n\n"
                        f"\u2022 **\u0916\u0930\u094d\u0947\u0926\u0940 \u0915\u093f\u0902\u092e\u0924 (Goods Cost)**: Rs {cog:,}\n"
                        f"\u2022 **\u0907\u0924\u0930 \u0916\u0930\u094d\u091a (OPEX)**: Rs {opex:,}\n"
                        f"\u2022 **\u090f\u0915\u0942\u0923 \u0916\u0930\u094d\u091a (Total Cost)**: Rs {total_cost:,}\n\n"
                        f"**\u0938\u0932\u094d\u0932\u093e**:\n"
                        f"\u2022 **\u0938\u094c\u091a\u0935\u0932\u094d\u0947\u0932\u0940 \u0935\u093f\u0915\u094d\u0930\u0940 \u0915\u093f\u0902\u092e\u0924**: Rs {suggested:.0f} (\u092f\u093e\u0935\u0930 \u096b\u096b% \u0928\u092b\u093e \u092e\u093f\u0933\u094d\u0947\u0932).\n"
                        f"\u2022 **\u090a\u0926\u094d\u0926\u093f\u0937\u094d\u091f \u0935\u093f\u0915\u094d\u0930\u0940 \u0915\u093f\u0902\u092e\u0924 (Target Price)**: Rs {target:.0f}\n"
                        f"\u2022 **\u0915\u093f\u092e\u093e\u0928 \u0935\u093f\u0915\u094d\u0930\u0940 \u0915\u093f\u0902\u092e\u0924 (Breakeven)**: Rs {breakeven:.0f} (\u092f\u093e\u092a\u094d\u0947\u0915\u094d\u0937\u093e \u0915\u092e\u0940 \u0915\u093f\u0902\u092e\u0924\u0940\u0932\u093e \u0935\u093f\u0915\u0942 \u0928\u0915\u093e).\n\n"
                        f"**\u091f\u094d\u0940\u092a**: \u091c\u0935\u0933\u091a\u094d\u092f\u093e \u0938\u094d\u092a\u0930\u094d\u0927\u0915\u093e\u0902\u0935\u0930 \u0932\u0915\u094d\u0937 \u0920\u094d\u0947\u0935\u093e.")
            else:
                return (f"Here is your suggested pricing analysis:\n\n"
                        f"\u2022 **Cost of Goods**: Rs {cog:,}\n"
                        f"\u2022 **Operating Expenses**: Rs {opex:,}\n"
                        f"\u2022 **Total Cost**: Rs {total_cost:,}\n\n"
                        f"**Pricing Options**:\n"
                        f"\u2022 **Suggested Price**: Rs {suggested:.0f} (Includes ~22% average retail markup).\n"
                        f"\u2022 **Target Price (at {margin:.0f}% margin)**: Rs {target:.0f}\n"
                        f"\u2022 **Breakeven (Minimum Price)**: Rs {breakeven:.0f} (Do not sell below this to avoid loss).\n\n"
                        f"**Tip**: Offer digital payment discounts to attract younger customers.")
        else:
            if language == "hi":
                return "Apne product ka cost (kharid daam) aur operating cost batayein, main price calculate kar dunga."
            elif language == "mr":
                return "\u0915\u0943\u092a\u092f\u093e \u0924\u094c\u092e\u091a\u094d\u092f\u093e \u0909\u0924\u094d\u092a\u093e\u0926\u0928\u093e\u091a\u0940 \u0916\u0930\u094d\u0947\u0926\u0940 \u0915\u093f\u0902\u092e\u0924 \u0938\u093e\u0902\u0917\u093e, \u092e\u0940 \u0935\u093f\u0915\u094d\u0930\u0940 \u0915\u093f\u0902\u092e\u0924 \u0920\u0930\u0935\u0942\u0928 \u0926\u094d\u0947\u0908\u0928."
            else:
                return "Please share the cost of goods and operating expenses to calculate the suggested selling price."

    # C. TAX / GST INTENT
    if intent == "tax_query" or "gst" in msg_lower or "turnover" in msg_lower or "tax" in msg_lower:
        import re
        turnover_nums = re.findall(r'\b\d+(?:\s*lakh|\s*l)?\b', msg_lower)
        turnover = 2500000.0  # default
        if turnover_nums:
            txt_num = turnover_nums[0]
            if "lakh" in txt_num or "l" in txt_num:
                digits = re.findall(r'\d+', txt_num)
                if digits:
                    turnover = float(digits[0]) * 100000
            else:
                turnover = float(re.findall(r'\d+', txt_num)[0])

        if "gst" in msg_lower or "composition" in msg_lower:
            mandatory = 4000000.0
            composition = 3000000.0
            
            if turnover < mandatory:
                status_en = "NOT REQUIRED"
                status_hi = "ZAROORI NAHI HAI"
                status_mr = "\u0906\u0935\u0936\u094d\u092f\u0915 \u0928\u093e\u0939\u0940"
                desc_en = f"Your annual turnover is Rs {turnover:,}, which is below the Rs 40 Lakh mandatory GST threshold for goods. No registration needed!"
                desc_hi = f"Aapka salana turnover Rs {turnover:,} hai, jo goods ke liye Rs 40 Lakh ki mandatory limit se kam hai. GST registration ki koi zarurat nahi hai!"
                desc_mr = f"\u0924\u094c\u092e\u091a\u0940 \u0935\u093e\u0930\u094d\u0937\u093f\u0915 \u090a\u0932\u093e\u0922\u093e\u0932 Rs {turnover:,} \u0906\u0939\u094d\u092f\u0947, \u091c\u0940 \u092e\u093e\u0932 \u0935\u094d\u092f\u093e\u092a\u093e\u0930\u093e\u0938\u093e\u0920\u0940 Rs \u096a\u096e \u0932\u093e\u0916\u093e\u0902\u091a\u094d\u092f\u093e \u092e\u0930\u094d\u092f\u093e\u0926\u094d\u0947\u092a\u094d\u0937\u093e \u0915\u092e\u0940 \u0906\u0939\u094d\u092f\u0947. \u091c\u094d\u092f\u0940\u090f\u0938\u091f\u094d\u0940 \u0928\u094b\u0902\u0926\u0923\u0940\u091a\u0940 \u0917\u0930\u091c \u0928\u093e\u0939\u0940!"
            else:
                status_en = "MANDATORY"
                status_hi = "ZAROORI HAI"
                status_mr = "\u0905\u0928\u093f\u0935\u093e\u0930\u094d\u092f \u0906\u0939\u094d\u092f\u0947"
                desc_en = f"Your annual turnover is Rs {turnover:,}, exceeding the exemption limit. You must apply for GST registration."
                desc_hi = f"Aapka turnover Rs {turnover:,} hai, jo exemption limit se zyada hai. Aapko GST registration karana padega."
                desc_mr = f"\u0924\u094c\u092e\u091a\u0940 \u090a\u0932\u093e\u0922\u093e\u0932 Rs {turnover:,} \u0906\u0939\u094d\u092f\u0947, \u091c\u0940 \u092e\u0930\u094d\u092f\u093e\u0926\u094d\u0947\u092a\u094d\u0937\u093e \u091c\u093e\u0938\u094d\u0924 \u0906\u0939\u094d\u092f\u0947. \u091c\u094d\u092f\u0940\u090f\u0938\u091f\u094d\u0940 \u0928\u094b\u0902\u0926\u0923\u0940 \u0915\u0930\u0923\u094d\u0947 \u0906\u0935\u0936\u094d\u092f\u0915 \u0906\u0939\u094d\u092f\u0947."

            if language == "hi":
                return (f"GST Registration status:\n\n"
                        f"\u2022 **GST Status**: **{status_hi}**\n"
                        f"\u2022 **Exemption limit**: Goods ke liye Rs 40 Lakh, Services ke liye Rs 20 Lakh.\n"
                        f"\u2022 **Aapka turnover**: Rs {turnover:,}\n\n"
                        f"**Salah**: {desc_hi} Faltu tax compliance se bachein. Agar turnover Rs 40 Lakh cross hota hai, toh Composition Scheme elect karein (jisme sirf 1% flat GST bharna hota hai).")
            elif language == "mr":
                return (f"\u091c\u094d\u092f\u0940\u090f\u0938\u091f\u094d\u0940 \u0928\u094b\u0902\u0926\u0923\u0940 \u092c\u093e\u092c\u0924 \u092e\u093e\u0930\u094d\u0917\u0926\u0930\u094d\u0936\u0928:\n\n"
                        f"\u2022 **\u091c\u094d\u092f\u0940\u090f\u0938\u091f\u094d\u0940 \u0928\u094b\u0902\u0926\u0923\u0940**: **{status_mr}**\n"
                        f"\u2022 **\u092e\u0930\u094d\u092f\u093e\u0926\u093e**: \u092e\u093e\u0932\u093e\u0938\u093e\u0920\u0940 Rs \u096a\u096e \u0932\u093e\u0916, \u0938\u094d\u0947\u0935\u093e\u0902\u0938\u093e\u0920\u0940 Rs \u096b\u096e \u0932\u093e\u0916.\n"
                        f"\u2022 **\u0924\u094c\u092e\u091a\u0940 \u090a\u0932\u093e\u0922\u093e\u0932**: Rs {turnover:,}\n\n"
                        f"**\u0938\u0932\u094d\u0932\u093e**: {desc_mr} \u0915\u0902\u092a\u094b\u091c\u093f\u0936\u0928 \u0938\u094d\u0915\u094d\u0940\u092e \u0928\u093f\u0935\u0921\u0942\u0928 \u0924\u094c\u092e\u094d\u0939\u0940 \u092b\u0915\u094d\u0924 \u0967% \u091c\u094d\u092f\u0940\u090f\u0938\u091f\u094d\u0940 \u092d\u0930\u0942 \u0936\u0915\u0924\u093e.")
            else:
                return (f"GST Registration Details:\n\n"
                        f"\u2022 **Status**: **{status_en}**\n"
                        f"\u2022 **Threshold limits**: Rs 40 Lakhs for Goods / Rs 20 Lakhs for Services.\n"
                        f"\u2022 **Your Turnover**: Rs {turnover:,}\n\n"
                        f"**Recommendation**: {desc_en} If you ever exceed Rs 40 Lakhs, apply for the Composition Scheme to pay a flat 1% GST and file simple quarterly returns.")

        # Income Tax
        if language == "hi":
            return ("Income Tax aur Slab Rules (FY 2023-24):\n\n"
                    "\u2022 Naye Tax Regime me **Rs 7 Lakh tak ki income par shunya (zero) tax** hai (u/s 87A rebate).\n"
                    "\u2022 Chhote shopkeepers and vendors ke liye **ITR-4** (Presumptive scheme) behtareen hai. Isme aapko turnover ka sirf 8% income declare karna hota hai (UPI/online payments par 6% declaration). No books or audit needed.\n\n"
                    "**Tip**: Har saal ITR bharein. ITR proof dikha kar bank se low-interest business loan aaram se mil jata hai.")
        elif language == "mr":
            # Marathi
            return ("\u0906\u092f\u0915\u0930 \u0906\u0923\u093f \u0906\u092f\u091f\u094d\u0940\u0906\u0930 \u092e\u093e\u0939\u093f\u0924\u0940:\n\n"
                    "\u2022 \u0928\u0935\u0940\u0928 \u0915\u0930 \u0930\u091a\u0928\u094d\u0947\u0928\u094c\u0938\u093e\u0930 **Rs \u096d \u0932\u093e\u0916\u093e\u0902\u092a\u0930\u094d\u092f\u0902\u0924\u091a\u094d\u092f\u093e \u0935\u093e\u0930\u094d\u0937\u093f\u0915 \u090a\u0925\u094d\u092a\u0928\u094d\u0928\u093e\u0935\u0930 \u0915\u094b\u0923\u0924\u093e\u0939\u0940 \u0915\u0930 \u0906\u0915\u093e\u0930\u0932\u093e \u091c\u093e\u0924 \u0928\u093e\u0939\u0940** (\u0915\u0932\u092e \u096e\u096dA \u0928\u094c\u0938\u093e\u0930).\n"
                    "\u2022 \u0932\u0918\u0942 \u0935\u094d\u092f\u093e\u092a\u093e\u0930\u094d\u094d\u092f\u093e\u0902\u0938\u093e\u0920\u0940 **ITR-4** \u0938\u0930\u094d\u0935\u094b\u092current \u092e\u093e\u0930\u094d\u0917 \u0906\u0939\u094d\u092f\u0947. \u0924\u094c\u092e\u094d\u0939\u0940 \u0916\u093e\u0924\u0940 \u0928 \u0920\u094d\u0947\u093check\u0924\u093e \u0924\u094c\u092e\u091a\u094d\u092f\u093e \u090f\u0915\u0942\u0923 \u090a\u0932\u093e\u0922\u093e\u0932\u094d\u0940\u091a\u094d\u092f\u093e \u092b\u0915\u094d\u0924 \u096e% (\u0921\u093f\u091c\u093f\u091f\u0932 \u0935\u094d\u092f\u0935\u0939\u093e\u0930\u093e\u0902\u0935\u0930 \u096a%) \u090a\u0925\u094d\u092a\u0928\u094d\u0928 \u0926\u093e\u0916\u0935\u0942 \u0936\u0915\u0924\u093e.\n\n"
                    "**\u091f\u094d\u0940\u092a**: \u0926\u0930\u0935\u0930\u094d\u0937\u0940 \u0906\u092f\u091f\u094d\u0940\u0906\u0930 \u0926\u093e\u0916\u0932 \u0915\u094d\u0947\u0932\u094d\u092f\u093e\u0938 \u092c\u094d\u0900\u0902\u0915 \u0915\u0930\u094d\u091c \u0938\u094c\u0932\u092d\u0924\u094d\u0947\u0928\u094d\u0947 \u092e\u093f\u0933\u094d\u0947\u0924\u094d\u0947.")
        else:
            return ("Income Tax & Filing Advice (FY 2023-24):\n\n"
                    "\u2022 Under the New Tax Regime, there is **zero tax for income up to Rs 7 Lakhs** (thanks to Section 87A tax rebate).\n"
                    "\u2022 For micro-businesses and shopkeepers, **ITR-4** (Presumptive Taxation Scheme u/s 44AD) is highly recommended. You can declare a flat 8% of your annual turnover as taxable income (reduced to 6% for digital receipts like UPI). No bookkeeping or audit required.\n\n"
                    "**Tip**: Even with zero tax, filing ITR regularly builds a strong credit record and helps you secure bank loans easily.")

    # D. DEBT / LOAN PAYOFF INTENT
    if intent == "debt_query":
        profile = state.rag.get_profile(user_id)
        debts = profile.get("debts", [])
        total_debt = profile.get("total_debt", 0)
        monthly_income = profile.get("monthly_income", 0)
        total_emi = profile.get("total_emi", 0)
        dti = profile.get("debt_to_income", 0)
        health_level = engine_result.get("health_level", "FAIR") if engine_result else "FAIR"

        if not debts:
            if language == "hi":
                return "Aapke profile me abhi koi loan nahi hai. Yeh ekdam badhiya baat hai! Apni income ka 20% emergency fund me save karein."
            elif language == "mr":
                return "\u0924\u094c\u092e\u091a\u094d\u092f\u093e \u0916\u093e\u0924\u094d\u092f\u093e\u0935\u0930 \u0938\u0927\u094d\u092f\u093e \u0915\u094b\u0923\u0924\u094d\u0947\u0939\u0940 \u0915\u0930\u094d\u091c \u0928\u093e\u0939\u0940. \u0939\u0940 \u0916\u0942\u092a \u091a\u093e\u0902\u0917\u0932\u0940 \u0917\u094b\u0937\u094d\u091f \u0906\u0939\u094d\u092f\u0947! \u096b\u096b% \u090a\u0925\u094d\u092a\u0928\u094d\u0928 \u092c\u091a\u0924\u0940\u092e\u0927\u094d\u092f\u0947 \u0917\u094c\u0902\u0924\u0935\u093e."
            else:
                return "You currently have no active debts registered! This is excellent. Consider saving 20% of your income in an emergency fund."

        avalanche_order = sorted(debts, key=lambda d: d.get("annual_rate", 0), reverse=True)
        priority_loan = avalanche_order[0]["name"]
        priority_rate = avalanche_order[0]["annual_rate"]
        extra_payment = max(1000.0, monthly_income * 0.1)

        if language == "hi":
            strategy_desc = "Hamara sujhaav: **Debt Avalanche** strategy apnayein (sabse zyada byaaj wala loan sabse pehle chukayein)."
            return (f"Aapke loans ka Debt Escape Roadmap:\n\n"
                    f"\u2022 **Total Karza**: Rs {total_debt:,}\n"
                    f"\u2022 **Monthly Income**: Rs {monthly_income:,}\n"
                    f"\u2022 **Total EMI Burden**: Rs {total_emi:,}\n"
                    f"\u2022 **Debt-to-Income (DTI) ratio**: {dti}% ({health_level} sthiti)\n\n"
                    f"**Suggested Strategy**: **{strategy_desc}**\n\n"
                    f"**Action Plan**:\n"
                    f"1. Sabhi loans par minimum EMI time par bharein taaki late fee na lage.\n"
                    f"2. Bacha hua extra cash (jaise har mahine Rs {extra_payment:,.0f}) sabse pehle **{priority_loan}** ({priority_rate}% byaaj) ko chukane me lagayein.\n\n"
                    f"**RBI Rules (Aapke Adhikar)**:\n"
                    f"\u2022 Bank/agent subah 8 se pehle aur raat 8 ke baad call nahi kar sakte. Agent badtameezi kare toh direct RBI Ombudsman me free me complaint karein.\n\n"
                    f"**Negotiation Tip**:\n"
                    f"Credit card bank ko settlement ke liye bole: *'Main severe financial hardship me hoon. Main 60% amount dekar one-time settlement karna chahta hoon. Mujhe senior recovery head se jodein.'*")
        elif language == "mr":
            # Marathi
            return (f"\u0915\u0930\u094d\u091c\u092e\u094c\u0915\u094d\u0924\u0940\u091a\u093e \u0906\u0930\u093e\u0916\u0921\u093e (Debt Roadmap):\n\n"
                    f"\u2022 **\u090f\u0915\u0942\u0923 \u0915\u0930\u094d\u091c**: Rs {total_debt:,}\n"
                    f"\u2022 **\u092e\u093e\u0938\u093f\u0915 \u090a\u0925\u094d\u092a\u0928\u094d\u0928**: Rs {monthly_income:,}\n"
                    f"\u2022 **\u092e\u093e\u0938\u093f\u0915 \u0939\u092a\u094d\u0924\u093e (EMI)**: Rs {total_emi:,}\n"
                    f"\u2022 **\u0915\u0930\u094d\u091c-\u090a\u0925\u094d\u092a\u0928\u094d\u0928 \u092a\u094d\u0930\u092e\u093e\u0923 (DTI)**: {dti}%\n\n"
                    f"**\u0938\u0932\u094d\u0932\u093e**: **Debt Avalanche** (\u0938\u0930\u094d\u0935\u094b\u092current \u0935\u094d\u092f\u093e\u091c\u0926\u0930 \u0906\u0927\u0940 \u092b\u094d\u0947\u0921\u0923\u094d\u0947) \u0939\u0940 \u092a\u0926\u094d\u0927\u0924 \u0935\u093e\u092a\u0930\u093e.\n\n"
                    f"**\u0915\u0943\u0924\u094d\u091f\u094d\u092f \u092f\u094b\u091c\u0928\u093e**:\n"
                    f"१. \u0938\u0930\u094d\u0935 \u0915\u0930\u094d\u091c\u093e\u0902\u091a\u094d\u0947 \u092e\u093e\u0938\u093f\u0915 \u0939\u092a\u094d\u0924\u094d\u0947 \u0935\u094d\u0947\u093check\u094d\u0924 \u092d\u0930\u093e.\n"
                    f"२. \u0905\u0924\u093f\u0930\u093f\u0915\u094d\u0924 \u0930\u0915\u094d\u0915\u092e \u0938\u0930\u094d\u0935\u093e\u0924 \u0906\u0927\u0940 **{priority_loan}** (\u0935\u094d\u092f\u093e\u091c\u0926\u0930 {priority_rate}%) \u092b\u094d\u0947\u0921\u0923\u094d\u0947\u0938\u093e\u0920\u0940 \u0935\u093e\u092a\u0930\u093e.\n\n"
                    f"**\u0924\u094c\u092e\u091a\u094d\u092f\u093e \u0938\u093e\u0920\u0940 \u0928\u093f\u092f\u092e**:\n"
                    f"\u2022 \u0935\u0938\u094c\u0932\u0940 \u090f\u091c\u0902\u091f \u0938\u0915\u093e\u0933\u0940 \u096e \u092a\u0942\u0930\u094d\u0935\u0940 \u0935 \u0930\u093e\u0924\u094d\u0930\u0940 \u096e \u0928\u0902\u0924\u0930 \u0915\u094c\u0932 \u0915\u0930\u0942 \u0936\u0915\u0924 \u0928\u093e\u0939\u0940\u0924. \u0924\u0915\u094d\u0930\u093e\u0930\u0940\u0938\u093e\u0920\u0940 RBI Ombudsman \u0915\u0921\u094d\u0947 \u0924\u0915\u094d\u0930\u093e\u0930 \u0915\u0930\u093e.")
        else:
            return (f"Debt Payoff Strategy & Rights:\n\n"
                    f"\u2022 **Total Debt**: Rs {total_debt:,}\n"
                    f"\u2022 **Monthly Income**: Rs {monthly_income:,}\n"
                    f"\u2022 **Monthly EMI Burden**: Rs {total_emi:,}\n"
                    f"\u2022 **Debt-to-Income (DTI)**: {dti}% ({health_level} status)\n\n"
                    f"**Recommended Approach**: **Debt Avalanche** (pay high-interest rate loans first).\n\n"
                    f"**Priority Order Plan**:\n"
                    f"1. Pay the minimum EMIs on all active loans.\n"
                    f"2. Put any extra monthly cash (e.g., Rs {extra_payment:,.0f}) toward **{priority_loan}** (charging {priority_rate}% interest).\n\n"
                    f"**Your Rights (RBI Rules)**:\n"
                    f"\u2022 Agents cannot contact you before 8 AM or after 8 PM. Harassment is illegal. File a free complaint online with the RBI Ombudsman.\n\n"
                    f"**Restructuring / Settlement Tip**:\n"
                    f"For credit cards, ask the bank: *'I am facing genuine financial hardship. I want to request a one-time settlement at 60% of outstanding. Please refer me to the settlement officer.'*")

    # E. KNOWLEDGE BASE MATCHES (MUDRA / RIGHTS)
    if "mudra" in msg_lower:
        if language == "hi":
            return ("Pradhan Mantri MUDRA Loan ki jaankari:\n\n"
                    "\u2022 **Mudra Shishu Loan**: Rs 50,000 tak, byaaj lagbhag 10-12% salana. Isme koi collateral (guarantee) nahi chahiye. Street vendors aur chhote shops ke liye best hai.\n"
                    "\u2022 **Mudra Kishor Loan**: Rs 50,000 se Rs 5 Lakh tak.\n"
                    "\u2022 **Tarun Loan**: Rs 5 Lakh se Rs 10 Lakh tak.\n\n"
                    "**Kahan apply karein?**: Kisi bhi sarkari bank (jaise SBI, PNB, Bank of Baroda) ya NBFC branch me jaakar baat karein. mudra.org.in par online bhi register kar sakte hain.\n\n"
                    "**Documents Checklist**:\n"
                    "1. Aadhaar Card aur PAN Card\n"
                    "2. Pichhle 3 mahine ka bank statement\n"
                    "3. Dukan/Business ka board ya outline ke sath photo\n"
                    "4. Address proof (rent agreement ya utility bill)")
        elif language == "mr":
            # Marathi
            return ("\u092a\u094d\u0930\u0927\u093e\u0928\u092e\u0902\u0924\u094d\u0930\u0940 \u092e\u094c\u0926\u094d\u0930\u093e (MUDRA) \u0915\u0930\u094d\u091c \u092f\u094b\u091c\u0928\u093e:\n\n"
                    "\u2022 **\u0936\u093f\u0936\u0942 \u0915\u0930\u094d\u091c**: Rs \u096b\u096e,\u096e\u096e\u096e \u092a\u0930\u094d\u092f\u0902\u0924, \u0935\u094d\u092f\u093e\u091c\u0926\u0930 \u0967\u096e-\u0967\u096a%. \u0915\u094b\u0923\u0924\u094d\u0940\u0939\u0940 \u0939\u092e\u0940 \u0915\u093f\u0902\u0935\u093e \u0924\u093e\u0930\u0923 \u0906\u0935\u0936\u094d\u092f\u0915 \u0928\u093e\u0939\u0940.\n"
                    "\u2022 **\u0915\u093f\u0936\u094b\u0930 \u0915\u0930\u094d\u091c**: Rs \u096b\u096e,\u096e\u096e\u096e \u0924\u094d\u0947 Rs \u096b \u0932\u093e\u0916.\n"
                    "\u2022 **\u0924\u0930\u094d\u0942\u0923 \u0915\u0930\u094d\u091c**: Rs \u096b \u0932\u093e\u0916 \u0924\u094d\u0947 Rs \u0967\u096e \u0932\u093e\u0916.\n\n"
                    "**\u0915\u094c\u0920\u094d\u0947 \u0905\u0930\u094d\u091c \u0915\u0930\u093e\u0935\u093e?**: \u091c\u0935\u0933\u091a\u094d\u092f\u093e \u0938\u0930\u0915\u093e\u0930\u0940 \u092c\u094d\u0900\u0902\u0915\u094d\u0947\u0924 \u0915\u093f\u0902\u0935\u093e NBFC \u092e\u0927\u094d\u092f\u0947 \u091c\u093e\u090a\u0928 \u091a\u094c\u0915\u0936\u0940 \u0915\u0930\u093e \u0915\u093f\u0902\u0935\u093e mudra.org.in \u0935\u0930 \u0906\u0930\u094d\u091c \u0915\u0930\u093e.\n\n"
                    "**\u0915\u093e\u0917\u0926\u092a\u0924\u094d\u0930\u094d\u0947**:\n"
                    "\u0967. \u0906\u0927\u093e\u0930 \u0915\u093e\u0930\u094d\u0921 \u0906\u0923\u093f \u092a\u094d\u090e\u0928 \u0915\u093e\u0930\u094d\u0921\n"
                    "\u096a. \u0969 \u092e\u0939\u093f\u0928\u094d\u092f\u093e\u0902\u091a\u094d\u0947 \u092c\u094d\u0900\u0902\u0915 \u0938\u094d\u091f\u094d\u0947\u091f\u092e\u094d\u0947\u0928\u094d\u091f\n"
                    "\u0969. \u0935\u094d\u092f\u0935\u0938\u093e\u092f\u093e\u091a\u093e \u092b\u094b\u091f\u094b \u0935 \u092a\u0924\u094d\u0924\u093e \u092a\u094d\u0930\u092e\u093e\u0923")
        else:
            return ("Pradhan Mantri MUDRA Loan Scheme Guidelines:\n\n"
                    "\u2022 **Mudra Shishu Loan**: Up to Rs 50,000. Interest rates range from 10-12% per year. No collateral or third-party guarantee required. Ideal for roadside vendors and micro-shop owners.\n"
                    "\u2022 **Mudra Kishor Loan**: Rs 50,000 to Rs 5 Lakhs.\n"
                    "\u2022 **Tarun Loan**: Rs 5 Lakhs to Rs 10 Lakhs.\n\n"
                    "**How to Apply**: Visit any Public Sector Bank (like SBI, Bank of Baroda) or registered NBFC. You can also apply online via the Udyami Mitra portal at mudra.org.in.\n\n"
                    "**Required Documents**:\n"
                    "1. Aadhaar Card + PAN Card\n"
                    "2. 2 Passport size photos\n"
                    "3. Last 3 months bank statements/passbook\n"
                    "4. Business address proof and shop photo")

    if any(w in msg_lower for w in ["rights", "ombudsman", "call", "agent", "harass", "dharm", "police", "raat", "night"]):
        if language == "hi":
            return ("Bank Recovery Agent aur Rights (RBI Rules):\n\n"
                    "\u2022 **Timing Rule**: Bank ya recovery agents subah 8 baje se pehle aur raat 8 baje ke baad call ya visit nahi kar sakte. Raat ko call karna illegal hai.\n"
                    "\u2022 **Identity Proof**: Agent ko aane par apna official ID card aur authorization letter dikhana hoga.\n"
                    "\u2022 **Complain (RBI Ombudsman)**: Agar agents badtameezi karein ya dhamkayein, toh seedhe **RBI Banking Ombudsman** (cms.rbi.org.in) par online complaint karein. Yeh bilkul free hai aur kisi vakeel ki zarurat nahi hoti.\n\n"
                    "**Tip**: Agent se baat karte waqt phone recording chalu rakhein aur unhe RBI rules yaad dilayein.")
        elif language == "mr":
            # Marathi
            return ("\u0935\u0938\u094c\u0932\u0940 \u090f\u091c\u0902\u091f \u0935\u093f\u0930\u094d\u0927 \u0924\u094c\u092e\u091a\u094d\u092f\u093e \u0939\u0915\u094d\u0915\u093e\u0902\u091a\u0940 \u092e\u093e\u0939\u093f\u0924\u0940:\n\n"
                    "\u2022 **\u0935\u094d\u0947\u093check\u094d\u091a\u094d\u0947 \u092c\u0902\u0927\u0928**: \u092c\u094d\u0902\u0915 \u0915\u093f\u0902\u0935\u093e \u0930\u093f\u0915\u0935\u094d\u0939\u0930\u094d\u091f \u090f\u091c\u0902\u091f \u0938\u0915\u093e\u0933\u0940 \u096e \u092a\u0942\u0930\u094d\u0935\u0940 \u0935 \u0930\u093e\u0924\u094d\u0930\u0940 \u096e \u0928\u0902\u0924\u0930 \u0915\u094c\u0932 \u0915\u0930\u0942 \u0936\u0915\u0924 \u0928\u093e\u0939\u0940\u0924.\n"
                    "\u2022 **\u0924\u0915\u094d\u0930\u093e\u0930**: \u091b\u0933\u0935\u0923\u0942\u0915 \u091c\u093e\u0932\u094d\u094f \u0905\u0938\u0932\u094d\u094d\u092f\u093e\u0938 **RBI Banking Ombudsman** \u0915\u0921\u094d\u0947 (cms.rbi.org.in) \u0935\u0930 \u0924\u0915\u094d\u0930\u093e\u0930 \u0915\u0930\u093e.\n\n"
                    "**\u091f\u094d\u0940\u092a**: \u090f\u091c\u0902\u0924\u093e\u0936\u0940 \u092c\u094b\u0932\u0924\u093e\u0928\u093e \u0915\u094c\u0932 \u0930\u094d\u0947\u0915\u094d\u094a\u0930\u094d\u0921\u093f\u0902\u0917 \u0915\u0930\u0942\u0928 \u0920\u094d\u0947\u093check\u093e.")
        else:
            return ("Your Rights against Recovery Agents (RBI Fair Practices Code):\n\n"
                    "\u2022 **Calling Hours**: Agents are legally forbidden from calling or visiting you before 8 AM or after 8 PM.\n"
                    "\u2022 **Harassment**: Physical or mental harassment, abusive language, or threatening calls are strict violations of RBI guidelines.\n"
                    "\u2022 **Free Resolution (RBI Ombudsman)**: If a lender violates these codes, file a complaint on the RBI Complaint Management System (cms.rbi.org.in). The ombudsman is free, online, and does not require a lawyer.\n\n"
                    "**Tip**: Always record calls with recovery agents and request their official employee ID and bank authorization letter before discussing anything.")

    # F. MULTILINGUAL RAG FACT EXPANSION
    if rag_docs:
        best_doc = rag_docs[0]["text"]
        
        if language == "hi":
            return (f"Namaste! Mujhe aapki help karne me khushi hogi. RAG database facts ke anusar:\n\n"
                    f"{best_doc}\n\n"
                    f"Agar aapko loan escape roadmap chahie, toh Profile tab me loans add karke mujhse poochein. Main aapko details me plan bataunga.")
        elif language == "mr":
            return (f"\u0928\u092e\u0938\u094d\u0915\u093e\u0930! \u092e\u0932\u093e \u0924\u094c\u092e\u094d\u0939\u094d\u092f\u093e \u092e\u093e\u0930\u094d\u0917\u0926\u0930\u094d\u0936\u0928\u093e\u0938\u093e\u0920\u0940 \u092f\u0947\u0925\u094d\u0947 \u0906\u0928\u0902\u0926 \u0906\u0939\u094d\u092f\u0947. \u0906\u092e\u091a\u094d\u092f\u093e \u092e\u093e\u0939\u093f\u0924\u0940\u0928\u094c\u0938\u093e\u0930:\n\n"
                    f"{best_doc}\n\n"
                    f"\u0905\u0927\u093f\u0915 \u092e\u093e\u0930\u094d\u0917\u0926\u0930\u094d\u0936\u0928\u093e\u0938\u093e\u0920\u0940 \u0924\u094c\u092e\u091a\u094d\u092f\u093e \u0915\u0930\u094d\u091c\u093e\u091a\u0940 \u092e\u093e\u0939\u093f\u0924\u0940 \u092a\u094d\u0930\u094b\u092b\u093e\u0908\u0932 \u091f\u094d\u094d\u092f\u093e\u092c\u092e\u0927\u094d\u092f\u0947 \u0928\u094b\u0902\u0926\u0935\u093e.")
        else:
            return (f"Hello! I am happy to help. According to the official financial regulations:\n\n"
                    f"{best_doc}\n\n"
                    f"If you need a custom debt escape strategy or transaction tracking, please update your details in the 'Profile & Memory' tab and let me know.")

    # G. GENERAL FINANCIAL DEFINITIONS (EMI, ITR, TAX, etc.)
    if "emi" in msg_lower or "equated monthly installment" in msg_lower:
        if language == "hi":
            return ("**EMI (Equated Monthly Installment) kya hai?**\n\n"
                    "EMI ka matlab hota hai ki aapko bank ya lender ko har mahine ek nishchit (fixed) amount chukani hoti hai jab tak aapka loan poora na ho jaye. Isme do hisse hote hain:\n"
                    "1. **Principal (Mool dhan)**: Jo paisa aapne borrow kiya tha.\n"
                    "2. **Interest (Byaaj)**: Loan lene ke badle bank ka charge.\n\n"
                    "**Mahatvapoorna Formula**:\n"
                    "\u2022 **EMI = [P x R x (1+R)^N] / [(1+R)^N - 1]**\n"
                    "*(Jahan P = Loan Principal, R = Monthly Interest Rate, N = Tenure in months)*\n\n"
                    "**ArthaSathi Tips**:\n"
                    "\u2022 **EMI Time par bharein**: Late fees se bachein aur CIBIL score badhayein.\n"
                    "\u2022 **Extra Principal Prepayments**: Agar aap saal me sirf ek extra EMI barabar prepayment karte hain, toh aapka loan tenure 2-3 saal kam ho sakta hai!")
        elif language == "mr":
            return ("**EMI (Equated Monthly Installment) म्हणजे काय?**\n\n"
                    "EMI म्हणजे तुम्हाला बँकेला किंवा सावकाराला दरमहा द्यावा लागणारा एक निश्चित हप्ता. यामध्ये दोन भाग असतात:\n"
                    "१. **मुद्दल (Principal)**: तुम्ही घेतलेले मूळ कर्ज.\n"
                    "२. **व्याज (Interest)**: कर्जावर आकारले जाणारे व्याज.\n\n"
                    "**नियम आणि टीप**:\n"
                    "\u2022 वेळेवर ईएमआय भरल्यास सिबिल (CIBIL) स्कोर चांगला राहतो आणि भविष्यात सहज कर्ज मिळते.\n"
                    "\u2022 शक्य असल्यास दरवर्षी कर्जाची अतिरिक्त मुद्दल फेडा, ज्यामुळे तुमचे कर्ज लवकर संपेल.")
        else:
            return ("**What is an EMI (Equated Monthly Installment)?**\n\n"
                    "An EMI is a fixed payment made by a borrower to a lender at a specified date each calendar month. EMIs consist of two components:\n"
                    "1. **Principal**: The original amount borrowed from the lender.\n"
                    "2. **Interest**: The cost charged by the lender for borrowing the money.\n\n"
                    "**Mathematical Formula**:\n"
                    "\u2022 **EMI = [P x R x (1+R)^N] / [(1+R)^N - 1]**\n"
                    "*(Where P = Principal, R = Monthly Interest Rate, N = Number of monthly installments)*\n\n"
                    "**ArthaSathi Advice**:\n"
                    "\u2022 **Boost Credit Score**: Always pay your EMIs on time to keep a clean CIBIL report.\n"
                    "\u2022 **Principal Prepayments**: Making just one extra EMI prepayment per year can slash your loan tenure by up to 2-3 years and save thousands in interest!")

    return None


async def generate_response(user_message: str, user_id: str,
                              language: str = "hi",
                              intent: Optional[str] = None,
                              engine_result: Optional[Dict] = None) -> str:
    if state.rag is None:
        return _template_response(user_message, language)

    rag_docs = state.rag.store.search(user_message, k=3)
    smart_response = _generate_smart_response(user_message, user_id, language, intent, engine_result, rag_docs)
    if smart_response:
        return smart_response

    context = state.rag.get_context(user_id, user_message, engine_result, language)

    system_prompts = {
        "hi":  "Tum ArthaSathi ho. Ek financial dost. Simple Hindi mein baat karo. Numbers clearly batao.",
        "en":  "You are ArthaSathi, a trusted financial companion. Be clear, specific, and honest.",
        "mr":  "\u0924\u094c\u092e\u094d\u0939\u094d\u092f\u093e \u092e\u093e\u0930\u094d\u0917\u0926\u0930\u094d\u0936\u0928\u093e\u0938\u093e\u0920\u0940 ArthaSathi.",
    }
    sys_prompt = system_prompts.get(language, system_prompts["en"])
    if context:
        sys_prompt += f"\n\n{context}"

    history = state.rag.memory.get_history(user_id, last_n=4)
    full_prompt = f"<|system|>\n{sys_prompt}[EOS]\n"
    for msg in history:
        full_prompt += f"<|{msg['role']}|>\n{msg['content']}[EOS]\n"
    full_prompt += f"<|user|>\n{user_message}[EOS]\n<|assistant|>\n"

    if state.model is None or state.tokenizer is None:
        return _template_response(user_message, language)

    try:
        import torch
        input_ids = state.tokenizer.encode(full_prompt, add_special_tokens=False).ids
        max_ctx = getattr(state.model.config, "context_length", 512)
        input_ids = input_ids[-max_ctx:]
        x = torch.tensor([input_ids], dtype=torch.long).to(cfg.DEVICE)

        with torch.inference_mode():
            output = state.model.generate(
                x,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                temperature=cfg.TEMPERATURE,
                top_k=cfg.TOP_K,
                top_p=cfg.TOP_P,
            )

        new_tokens = output[0, len(input_ids):].tolist()
        response = state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not response or len(response) < 10 or response.count(response[:4]) > 3:
            return _template_response(user_message, language)
        return response
    except Exception as e:
        print(f"[Inference] Error: {e}")
        return _template_response(user_message, language)

def _template_response(user_message: str, language: str) -> str:
    templates = {
        "hi": ("Namaste! Main ArthaSathi hoon — aapka financial dost. "
               "Main aapke loan, debt, aur business ke baare mein madad kar sakta hoon. "
               "Kripya apna sawaal detail mein batayein — "
               "jaise loan kitna hai, interest rate kya hai, monthly income kitni hai."),
        "en": ("Hello! I'm ArthaSathi, your financial companion. "
               "I can help with loans, debt management, business pricing, and tax guidance. "
               "Please share details like loan amount, interest rate, and monthly income."),
        "mr": ("\u0928\u092e\u0938\u094d\u0915\u093e\u0930! \u092e\u0940 ArthaSathi \u0906\u0939\u094d\u092f\u0947 \u093e \u0924\u094c\u092e\u091a\u093e \u0906\u0930\u094d\u0925\u093f\u0915 \u092e\u093f\u0924\u094d\u0930."),
    }
    return templates.get(language, templates["en"])

async def process_message(user_id: str, text: str,
                           channel: str = "whatsapp") -> str:
    if state.rag is None:
        return _template_response(text, "hi")

    language = state.rag.detect_language(text)
    user = state.rag.memory.get_or_create_user(user_id, language)

    engine_result = None
    intent = "general"
    try:
        profile_data = state.rag.get_profile(user_id)
        debts_raw = profile_data.get("debts", [])
        monthly_income = profile_data.get("monthly_income", 0)

        debts = [
            Debt(name=d["name"], principal=d["principal"],
                 annual_rate=d["annual_rate"], min_payment=d["min_payment"],
                 lender_type=d.get("lender_type", "bank"))
            for d in debts_raw
        ]
        up = UserFinancialProfile(
            user_id=user_id,
            monthly_income=monthly_income,
            debts=debts,
            language=language,
        )
        engine_ctx = state.advisor.process_message(text, language, up)
        engine_result = engine_ctx.get("engine_result")
        intent = engine_ctx.get("intent", "general")

        txn = state.advisor.business_engine.parse_voice_transaction(text, language)
        if txn and txn.get("confidence") == "high":
            state.rag.save_transaction(user_id, txn)
    except Exception as e:
        print(f"[Engine] Error: {e}")

    response = await generate_response(text, user_id, language, intent, engine_result)

    state.rag.save_user_message(user_id, text, language)
    state.rag.save_assistant_message(user_id, response, language)

    return response
# ─────────────────────────────────────────────────────────────
# 7. WHATSAPP WEBHOOK
# ─────────────────────────────────────────────────────────────

@app.get("/webhook")
async def wa_verify(request: Request):
    """WhatsApp webhook verification"""
    params = dict(request.query_params)
    mode   = params.get("hub.mode")
    token  = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == cfg.WA_VERIFY_TOKEN:
        print("[WhatsApp] Webhook verified [OK]")
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def wa_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive and process WhatsApp messages.
    Handles: text messages, voice notes (audio), reactions.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Parse WhatsApp Cloud API payload
    try:
        entry   = body["entry"][0]
        changes = entry["changes"][0]["value"]
        messages = changes.get("messages", [])

        for msg in messages:
            user_id  = msg["from"]
            msg_type = msg.get("type")

            if msg_type == "text":
                text = msg["text"]["body"]
                background_tasks.add_task(_handle_text_message, user_id, text)

            elif msg_type == "audio":
                audio_id = msg["audio"]["id"]
                background_tasks.add_task(_handle_voice_note, user_id, audio_id)

            elif msg_type == "interactive":
                # Handle button clicks
                reply_id = msg["interactive"]["button_reply"]["id"]
                background_tasks.add_task(_handle_button_reply, user_id, reply_id)

    except (KeyError, IndexError) as e:
        # Not a message event (could be status update, etc.)
        pass

    return JSONResponse({"status": "ok"})


async def _handle_text_message(user_id: str, text: str):
    """Process a text message from WhatsApp"""
    response = await process_message(user_id, text, "whatsapp")
    await send_whatsapp_message(user_id, response)


async def _handle_voice_note(user_id: str, audio_id: str):
    """
    Download and transcribe a WhatsApp voice note,
    then process and reply.
    """
    # Download audio from WhatsApp
    audio_bytes = await _download_wa_media(audio_id)
    if not audio_bytes:
        await send_whatsapp_message(user_id,
            "Voice note receive hua, lekin process nahi hua. Please text mein likhein.")
        return

    # Transcribe
    if state.voice:
        result   = state.voice.process_voice_note(audio_bytes)
        text     = result.get("text", "")
        language = result.get("language", "hi")
    else:
        await send_whatsapp_message(user_id,
            "Voice notes abhi available nahi. Please text mein likhein.")
        return

    if not text.strip():
        await send_whatsapp_message(user_id,
            "Voice note samajh nahi aaya. Dobara bolein ya text mein likhein.")
        return

    # Process transcribed text
    response = await process_message(user_id, text, "whatsapp")

    # Send text reply
    await send_whatsapp_message(user_id, response)

    # Optionally send audio reply (uncomment to enable)
    # if state.voice:
    #     audio_reply = state.voice.generate_audio_reply(response, language)
    #     if audio_reply:
    #         await send_whatsapp_audio(user_id, audio_reply)


async def _handle_button_reply(user_id: str, button_id: str):
    """Handle interactive button clicks"""
    responses = {
        "debt_help":    "Apne saare loans ka naam, amount aur interest rate batao.",
        "business_help": "Apne business ke baare mein batao — kya bechte ho, kitna kharcha hai?",
        "tax_help":     "Aapki saalana aur maheena kamaai kitni hai?",
    }
    text = responses.get(button_id, "Haan bolo, main sun raha hoon.")
    await send_whatsapp_message(user_id, text)


async def _download_wa_media(media_id: str) -> Optional[bytes]:
    """Download media from WhatsApp Cloud API"""
    if not cfg.WA_ACCESS_TOKEN:
        return None
    try:
        async with httpx.AsyncClient() as client:
            # Get media URL
            r = await client.get(
                f"https://graph.facebook.com/v18.0/{media_id}",
                headers={"Authorization": f"Bearer {cfg.WA_ACCESS_TOKEN}"},
                timeout=10
            )
            media_url = r.json().get("url")
            if not media_url:
                return None
            # Download media
            r2 = await client.get(
                media_url,
                headers={"Authorization": f"Bearer {cfg.WA_ACCESS_TOKEN}"},
                timeout=30
            )
            return r2.content
    except Exception as e:
        print(f"[WhatsApp] Media download failed: {e}")
        return None


async def send_whatsapp_message(to: str, text: str,
                                 with_buttons: bool = False) -> bool:
    """Send a text message via WhatsApp Cloud API"""
    if not cfg.WA_ACCESS_TOKEN or not cfg.WA_PHONE_ID:
        print(f"[WhatsApp->{to}] {text[:80]}...")
        return True  # Dev mode: just log

    # WhatsApp has 4096 char limit — split if needed
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]

    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type":    "individual",
                "to":                to,
                "type":              "text",
                "text":              {"preview_url": False, "body": chunk},
            }

            if with_buttons and len(chunks) == 1:
                # Add quick reply buttons for first interaction
                payload = {
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "interactive",
                    "interactive": {
                        "type": "button",
                        "body": {"text": chunk},
                        "action": {
                            "buttons": [
                                {"type":"reply","reply":{"id":"debt_help","title":"💳 Debt Help"}},
                                {"type":"reply","reply":{"id":"business_help","title":"🏪 Business"}},
                                {"type":"reply","reply":{"id":"tax_help","title":"📋 Tax/ITR"}},
                            ]
                        }
                    }
                }

            try:
                r = await client.post(
                    cfg.WA_API_URL.format(phone_id=cfg.WA_PHONE_ID),
                    headers={"Authorization": f"Bearer {cfg.WA_ACCESS_TOKEN}",
                             "Content-Type": "application/json"},
                    json=payload,
                    timeout=10,
                )
                if r.status_code != 200:
                    print(f"[WhatsApp] Send failed: {r.text}")
                    return False
            except Exception as e:
                print(f"[WhatsApp] Send error: {e}")
                return False
    return True


# ─────────────────────────────────────────────────────────────
# 8. SMS WEBHOOK (Twilio / MSG91 fallback)
# ─────────────────────────────────────────────────────────────

@app.post("/sms")
async def sms_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive SMS messages (Twilio webhook format).
    Used as fallback for users without WhatsApp.
    """
    form    = await request.form()
    from_no = form.get("From", "")
    body    = form.get("Body", "")

    if not from_no or not body:
        return PlainTextResponse("", status_code=200)

    background_tasks.add_task(_handle_sms, from_no, body)
    return PlainTextResponse("", status_code=200)


async def _handle_sms(from_no: str, text: str):
    """Process an incoming SMS"""
    response = await process_message(from_no, text, "sms")

    # SMS has 160 char limit — send multiple if needed
    chunks = _split_sms(response)
    for chunk in chunks[:5]:  # Max 5 SMS per reply
        await send_sms(from_no, chunk)


def _split_sms(text: str, limit: int = 155) -> List[str]:
    """Split text into SMS-sized chunks at word boundaries"""
    words  = text.split()
    chunks = []
    curr   = ""
    for w in words:
        if len(curr) + len(w) + 1 > limit:
            chunks.append(curr.strip())
            curr = w
        else:
            curr = (curr + " " + w).strip()
    if curr:
        chunks.append(curr)
    return chunks


async def send_sms(to: str, text: str) -> bool:
    """Send SMS via Twilio"""
    if not cfg.TWILIO_SID or not cfg.TWILIO_TOKEN:
        print(f"[SMS→{to}] {text[:80]}")
        return True  # Dev mode
    try:
        from twilio.rest import Client
        client = Client(cfg.TWILIO_SID, cfg.TWILIO_TOKEN)
        client.messages.create(body=text, from_=cfg.TWILIO_NUMBER, to=to)
        return True
    except ImportError:
        print("[SMS] pip install twilio")
        return False
    except Exception as e:
        print(f"[SMS] Send failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# 9. REST API ENDPOINTS
# ─────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str

@app.post("/register")
async def register(req: AuthRequest):
    if state.rag is None:
        raise HTTPException(503, "Service not ready")
    db = state.rag.memory.db
    row = db.execute("SELECT username FROM auth_users WHERE username=?", (req.username,)).fetchone()
    if row:
        raise HTTPException(400, "Username already exists")
    
    user_id = f"usr_{hashlib.md5(req.username.encode()).hexdigest()[:8]}"
    state.rag.memory.get_or_create_user(user_id)
    
    h = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
    db.execute("INSERT INTO auth_users VALUES (?, ?, ?)", (req.username, h, user_id))
    db.commit()
    return {"status": "registered", "user_id": user_id, "username": req.username}

@app.post("/login")
async def login(req: AuthRequest):
    if state.rag is None:
        raise HTTPException(503, "Service not ready")
    db = state.rag.memory.db
    h = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
    row = db.execute("SELECT user_id FROM auth_users WHERE username=? AND password_hash=?", 
                     (req.username, h)).fetchone()
    if not row:
        raise HTTPException(401, "Invalid username or password")
    
    return {"status": "authenticated", "user_id": row[0], "username": req.username}

class ChatRequest(BaseModel):
    user_id:  str
    message:  str
    language: str = "hi"

class DebtAddRequest(BaseModel):
    user_id:     str
    name:        str
    principal:   float
    annual_rate: float
    min_payment: float
    lender_type: str = "bank"

class TransactionRequest(BaseModel):
    user_id:    str
    type:       str       # income | expense
    amount:     float
    category:   str = "other"
    description: str = ""

class IncomeUpdateRequest(BaseModel):
    user_id:        str
    monthly_income: float


@app.get("/health")
async def health():
    return {
        "status":    "healthy" if state.ready else "starting",
        "model":     state.model is not None,
        "tokenizer": state.tokenizer is not None,
        "rag":       state.rag is not None,
        "voice":     state.voice is not None,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Direct REST API for chat — useful for app integration"""
    response = await process_message(req.user_id, req.message)
    return {"response": response, "user_id": req.user_id}


@app.post("/user/debt")
async def add_debt(req: DebtAddRequest):
    """Add a debt to user's profile"""
    if state.rag is None:
        raise HTTPException(503, "Service not ready")
    state.rag.add_debt(
        req.user_id, name=req.name, principal=req.principal,
        annual_rate=req.annual_rate, min_payment=req.min_payment,
        lender_type=req.lender_type
    )
    return {"status": "added", "debt_name": req.name}


@app.post("/user/transaction")
async def add_transaction(req: TransactionRequest):
    """Add a transaction to user's financial history"""
    if state.rag is None:
        raise HTTPException(503, "Service not ready")
    state.rag.memory.add_transaction(
        req.user_id, req.type, req.amount, req.category, req.description
    )
    return {"status": "recorded", "amount": req.amount, "type": req.type}


@app.post("/user/income")
async def update_income(req: IncomeUpdateRequest):
    """Update user's monthly income"""
    if state.rag is None:
        raise HTTPException(503, "Service not ready")
    state.rag.memory.update_user(req.user_id, monthly_income=req.monthly_income)
    return {"status": "updated", "monthly_income": req.monthly_income}


@app.get("/user/{user_id}/profile")
async def get_profile(user_id: str):
    """Get complete user financial profile"""
    if state.rag is None:
        raise HTTPException(503, "Service not ready")
    return state.rag.get_profile(user_id)


@app.post("/calculate/emi")
async def calculate_emi(principal: float, annual_rate: float, tenure_months: int):
    """EMI calculator endpoint"""
    return state.advisor.debt_engine.emi_calculator(principal, annual_rate, tenure_months)


@app.post("/calculate/tax")
async def calculate_tax(annual_income: float, regime: str = "new"):
    """Income tax calculator endpoint"""
    return state.advisor.tax_engine.calculate_tax(annual_income, regime)


@app.post("/calculate/gst")
async def calculate_gst(annual_turnover: float, supplies_type: str = "goods"):
    """GST registration decision endpoint"""
    return state.advisor.business_engine.gst_decision(annual_turnover,
                                                        supplies_type=supplies_type)


# ─────────────────────────────────────────────────────────────
# 10. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("ArthaSathi API Server")
    print("=" * 55)
    print("WhatsApp Webhook: POST /webhook")
    print("SMS Webhook:      POST /sms")
    print("REST Chat:        POST /chat")
    print("Health Check:     GET  /health")
    print("=" * 55 + "\n")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
