import os
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.tokenizer import train_arthasathi_tokenizer

def main():
    print("Preparing training data for fast tokenizer training...")
    data_dir = Path("tokenizer_training_data")
    data_dir.mkdir(exist_ok=True)
    
    # Let's create a representative sample file for each language with rich multilingual texts
    # combined with some synthetic financial phrases.
    languages = ["hi", "en", "mr", "ta", "kn", "bho", "as", "bn", "te"]
    
    # Multilingual basic dictionary and text templates to bootstrap the vocabulary
    templates = {
        "hi": ["मेरा नाम राम है।", "मुझे कर्ज चुकाना है।", "क्रेडिट कार्ड का ब्याज दर बहुत अधिक है।", "साहूकार का ब्याज दर गैरकानूनी है।", "कंपोजिशन स्कीम में सिर्फ एक प्रतिशत जीएसटी लगता है।", "मुद्रा लोन के लिए आधार और पैन की जरूरत होती है।", "ईएमआई का भुगतान समय पर करना चाहिए।", "ऋण चुकाने की रणनीति में हिमस्खलन और स्नोबॉल शामिल हैं।", "बचत खाता खोलना महत्वपूर्ण है।", "व्यवसाय का दैनिक खर्च और आमदनी ट्रैक करें।"],
        "en": ["My name is John.", "I need to repay my loan.", "Credit card interest rates are very high.", "The moneylender charges illegal interest rates.", "The composition scheme requires only one percent GST.", "Mudra loan requires Aadhaar and PAN cards.", "EMI payments should be made on time.", "Debt payoff strategies include avalanche and snowball.", "Saving money is very important for business expansion.", "Track daily business income and expenses regularly."],
        "mr": ["माझे नाव श्याम आहे.", "मला कर्ज परत करायचे आहे.", "क्रेडिट कार्डचे व्याजदर खूप जास्त आहेत.", "सावकाराचे व्याजदर बेकायदेशीर आहेत.", "जीएसटी रजिस्ट्रेशन करण्यासाठी उलाढाल किती असावी?", "मुद्रा लोनसाठी आधार आणि पॅन आवश्यक आहे.", "ईएमआय वेळेवर भरणे आवश्यक आहे.", "कर्जमुक्तीसाठी कोणती रणनीती वापरावी?", "व्यवसायाची दैनिक विक्री आणि खर्च नोंदवा.", "आर्थिक नियोजन करणे महत्त्वाचे आहे."],
        "ta": ["என் பெயர் ராம்.", "நான் என் கடனைத் திரும்பச் செலுத்த வேண்டும்.", "கிரெடிட் கார்டு வட்டி விகிதங்கள் மிக அதிகம்.", "வட்டிக்கு பணம் கொடுப்பவர்கள் சட்டவிரோதமான வட்டி வசூலிக்கிறார்கள்.", "முத்ரா கடனுக்கு ஆதார் மற்றும் பான் கார்டு தேவை.", "இஎம்ஐ தொகையை சரியான நேரத்தில் செலுத்த வேண்டும்.", "கடன் அடைக்கும் முறைகளில் அவலாஞ்ச் மற்றும் ஸ்னோபால் உள்ளன.", "தினசரி வரவு செலவு கணக்கை பதிவு செய்யுங்கள்.", "ஜிஎஸ்டி பதிவு செய்ய வேண்டுமா?", "சேமிப்பு பழக்கம் மிகவும் முக்கியம்."],
        "kn": ["ನನ್ನ ಹೆಸರು ಕೃಷ್ಣ.", "ನಾನು ಸಾಲವನ್ನು ಮರುಪಾವತಿಸಬೇಕು.", "ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಬಡ್ಡಿ ದರ ತುಂಬಾ ಹೆಚ್ಚಾಗಿದೆ.", "ಲೇವದೇವಿಗಾರರು ಕಾನೂನುಬಾಹಿರ ಬಡ್ಡಿ ವಿಧಿಸುತ್ತಾರೆ.", "ಮುದ್ರಾ ಸಾಲಕ್ಕೆ ಆಧಾರ್ ಮತ್ತು ಪಾನ್ ಕಾರ್ಡ್ ಬೇಕು.", "ಇಎಂಐ ಸರಿಯಾದ ಸಮಯಕ್ಕೆ ಪಾವತಿಸಬೇಕು.", "ದೈನಂದिन ವ್ಯಾಪಾರದ ಆದಾಯ ಮತ್ತು ಖರ್ಚುಗಳನ್ನು ಟ್ರ್ಯಾಕ್ ಮಾಡಿ.", "ಜಿಎಸ್ಟಿ ನೋಂದಣಿ ಅಗತ್ಯವೇ?", "ಸಾಲ ತೀರಿಸುವ ವಿಧಾನಗಳು ಯಾವುವು?", "ಉಳಿತಾಯ ಮಾಡುವುದು ಒಳ್ಳೆಯದು."],
        "te": ["నా పేరు రాము.", "నేను నా అప్పు తీర్చాలి.", "క్రెడిట్ కార్డ్ వడ్డీ రేట్లు చాలా ఎక్కువగా ఉన్నాయి.", "వడ్డీ వ్యాపారులు చట్టవిరుద్ధంగా వడ్డీ వసూలు చేస్తున్నారు.", "ముద్రా లోన్ కొరకు ఆధార్ మరియు పాన్ కార్డ్ అవసరం.", "ఈఎమ్ఐ సకాలంలో చెల్లించాలి.", "రోజువారీ వ్యాపార ఆదాయం మరియు ఖర్చులను నమోదు చేయండి.", "జీఎస్టీ రిజిస్ట్రేషన్ అవసరమా?", "అప్పులు త్వరగా తీర్చడం ఎలా?", "ఆర్థిక క్రమశిక్షణ ముఖ్యం."],
        "bn": ["আমার নাম রাম।", "আমাকে ঋণ পরিশোধ করতে হবে।", "ক্রেডিট কার্ডের সুদের হার খুব বেশি।", "মহাজনরা বেআইনি সুদের হার দাবি করে।", "মুদ্রা লোনের জন্য আধার ও প্যান কার্ড প্রয়োজন।", "ইএমআই সময়মতো দেওয়া উচিত।", "দৈনিক ব্যবসার আয় ও ব্যয় হিসাব রাখুন।", "জিএসটি রেজিস্ট্রেশন কখন দরকার?", "ঋণ পরিশোধের বিভিন্ন উপায় কী কী?", "টাকা জমানো অত্যন্ত জরুরি।"],
        "as": ["মোৰ নাম ৰাম।", "মই ঋণ পৰিশোধ কৰিব লাগিব।", "ক্ৰেডিট কাৰ্ডৰ সুতৰ হাৰ বহুত বেছি।", "মহাজনসকলে বেআইনীভাৱে সুত লয়।", "মুদ্ৰা ঋণৰ বাবে আধাৰ আৰু পেন কাৰ্ড লাগিব।", "ইএমআই সময়মতে পৰিশোধ কৰিব লাগে।", "দৈনিক ব্যৱসায়ৰ আয় আৰু ব্যয় হিচাপ কৰক।", "ঋণ মুক্ত হোৱাৰ উপায় কি?", "জিএছটি পঞ্জীয়ন কৰিব লাগিবনে?", "বচত কৰাটো খুবেই প্ৰয়োজন।"],
        "bho": ["हमार नाम राम बा।", "हमार कर्जा चुकावे के बा।", "क्रेडिट कार्ड के ब्याज दर बहुत बेसी बा।", "साहूकार गैरकानूनी ब्याज दर लेत बा।", "मुद्रा लोन खातिर आधार अउर पैन कार्ड चाहीं।", "ईएमआई समे पर भरे के चाहीं।", "रोजाना के धंधा के कमाई अउर खर्चा लिखल करीं।", "कर्जा से मुक्ति कइसे मिली?", "जीएसटी भरे के परी का?", "पइसा बचावल बहुत जरूरी बा।"]
    }

    # Let's generate a rich sample vocabulary by generating variations of texts
    for lang in languages:
        out_file = data_dir / f"{lang}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            # Write templates many times to bootstrap words
            for _ in range(500):
                for temp in templates[lang]:
                    f.write(temp + "\n")
            
            # Write numeric sequences for digit BPE representation
            for i in range(1000):
                f.write(f"loan Rs {i*50} min payment {i*5} EMI {i*10} interest {i%15} percent\n")
                
            # Write vocabulary builders
            f.write(" [DEBT] [INCOME] [EXPENSE] [LOAN] [INTEREST] [TAX] [GST] [EMI] [AMOUNT] [LANG:HI] [LANG:EN] [LANG:MR] [LANG:TA] [LANG:KN] [LANG:BHO] [LANG:AS] [LANG:BN] [LANG:TE]\n")

    print("Multilingual training files created successfully.")
    
    # Train the tokenizer
    data_files = [str(p) for p in data_dir.glob("*.txt")]
    train_arthasathi_tokenizer(
        data_files=data_files,
        save_dir="arthasathi_tokenizer",
        vocab_size=60000,
        min_freq=1
    )
    print("Fast BPE Tokenizer training completed!")

if __name__ == "__main__":
    main()
