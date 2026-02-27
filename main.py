"""
main.py
========
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
from fastapi.responses import JSONResponse, PlainTextResponse
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
    print("[ArthaSathi] All systems ready! 🚀")

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


# ─────────────────────────────────────────────────────────────
# 5. CORE LLM INFERENCE
# ─────────────────────────────────────────────────────────────

async def generate_response(user_message: str, user_id: str,
                              language: str = "hi",
                              engine_result: Optional[Dict] = None) -> str:
    """
    Generate a response using the ArthaSathi LLM + RAG context.
    Falls back to template response if model not loaded.
    """
    if state.rag is None:
        return _template_response(user_message, language)

    # Build RAG context
    context = state.rag.get_context(user_id, user_message, engine_result, language)

    # Build full prompt in ChatML format
    system_prompts = {
        "hi":  "Tum ArthaSathi ho. Ek financial dost. Simple Hindi mein baat karo. Numbers clearly batao.",
        "en":  "You are ArthaSathi, a trusted financial companion. Be clear, specific, and honest.",
        "mr":  "तुम्ही ArthaSathi आहात. सोप्या मराठीत उत्तर द्या. आकडे स्पष्टपणे सांगा.",
        "ta":  "நீங்கள் ArthaSathi. எளிய தமிழில் பதிலளிக்கவும். எண்களை தெளிவாக சொல்லுங்கள்.",
        "kn":  "ನೀವು ArthaSathi. ಸರಳ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
        "bho": "Raho ArthaSathi. Bhojpuri mein baat karo.",
        "as":  "আপুনি ArthaSathi. সহজ অসমীয়াত উত্তৰ দিয়ক।",
        "bn":  "তুমি ArthaSathi. সহজ বাংলায় উত্তর দাও।",
        "te":  "మీరు ArthaSathi. సరళమైన తెలుగులో సమాధానం చెప్పండి.",
    }
    sys_prompt = system_prompts.get(language, system_prompts["en"])

    if context:
        sys_prompt += f"\n\n{context}"

    # Get conversation history
    history = state.rag.memory.get_history(user_id, last_n=4)

    full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
    for msg in history:
        full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    full_prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    if state.model is None or state.tokenizer is None:
        return _template_response(user_message, language)

    try:
        import torch

        # Tokenize
        input_ids = state.tokenizer.encode(full_prompt).ids
        input_ids = input_ids[-800:]  # Trim to fit context window
        x = torch.tensor([input_ids], dtype=torch.long).to(cfg.DEVICE)

        # Generate
        with torch.inference_mode():
            output = state.model.generate(
                x,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                temperature=cfg.TEMPERATURE,
                top_k=cfg.TOP_K,
                top_p=cfg.TOP_P,
            )

        # Decode only new tokens
        new_tokens = output[0, len(input_ids):].tolist()
        response   = state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not response:
            return _template_response(user_message, language)

        return response

    except Exception as e:
        print(f"[Inference] Error: {e}")
        return _template_response(user_message, language)


def _template_response(user_message: str, language: str) -> str:
    """Template response when model is not yet trained"""
    templates = {
        "hi": ("Namaste! Main ArthaSathi hoon — aapka financial dost. "
               "Main aapke loan, debt, aur business ke baare mein madad kar sakta hoon. "
               "Kripya apna sawaal detail mein batayein — "
               "jaise loan kitna hai, interest rate kya hai, monthly income kitni hai."),
        "en": ("Hello! I'm ArthaSathi, your financial companion. "
               "I can help with loans, debt management, business pricing, and tax guidance. "
               "Please share details like loan amount, interest rate, and monthly income."),
        "mr": ("नमस्ते! मी ArthaSathi आहे — तुमचा आर्थिक मित्र. "
               "मी कर्ज, व्यवसाय आणि कर याबद्दल मदत करू शकतो."),
        "ta": ("வணக்கம்! நான் ArthaSathi — உங்கள் நிதி நண்பர். "
               "கடன், வணிகம், வரி குறித்து உதவ தயாராக இருக்கிறேன்."),
    }
    return templates.get(language, templates["en"])


# ─────────────────────────────────────────────────────────────
# 6. MESSAGE PROCESSOR (main logic for every incoming message)
# ─────────────────────────────────────────────────────────────

async def process_message(user_id: str, text: str,
                           channel: str = "whatsapp") -> str:
    """
    Core message processing pipeline:
    1. Detect language
    2. Update user record
    3. Parse financial intent
    4. Run relevant financial engine
    5. Generate LLM response with context
    6. Save to memory
    7. Return response text
    """
    if state.rag is None:
        return _template_response(text, "hi")

    # 1. Detect language
    language = state.rag.detect_language(text)

    # 2. Get/create user
    user = state.rag.memory.get_or_create_user(user_id, language)

    # 3. Parse intent + run financial engine
    engine_result = None
    try:
        profile_data  = state.rag.get_profile(user_id)
        debts_raw     = profile_data.get("debts", [])
        monthly_income = profile_data.get("monthly_income", 0)

        # Build domain profile
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

        # Auto-parse transaction from voice note text
        txn = state.advisor.business_engine.parse_voice_transaction(text, language)
        if txn and txn.get("confidence") == "high":
            state.rag.save_transaction(user_id, txn)

    except Exception as e:
        print(f"[Engine] Error: {e}")

    # 4. Generate response
    response = await generate_response(text, user_id, language, engine_result)

    # 5. Save to memory
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
        print("[WhatsApp] Webhook verified ✓")
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
        print(f"[WhatsApp→{to}] {text[:80]}...")
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
