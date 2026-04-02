# app/prompts/medical_safety.py
"""
Medical safety guardrails and compliance prompts.

These are NON-NEGOTIABLE — every response must pass through safety checks.
"""

# Keywords that trigger emergency escalation
EMERGENCY_KEYWORDS = [
    # English
    "chest pain", "heart attack", "stroke", "difficulty breathing",
    "shortness of breath", "severe bleeding", "unconscious", "suicide",
    "self harm", "self-harm", "overdose", "choking", "seizure",
    "anaphylaxis", "allergic reaction severe", "poisoning",
    # Hindi
    "सीने में दर्द", "हार्ट अटैक", "सांस लेने में तकलीफ",
    "बेहोश", "खून बह रहा", "आत्महत्या", "दौरा",
]

# Standard disclaimer appended to medical responses
DISCLAIMER = (
    "\n\n⚠️ *This is for informational purposes only and does not constitute "
    "medical advice. Please consult a qualified healthcare professional for "
    "proper diagnosis and treatment.*"
)

DISCLAIMER_HI = (
    "\n\n⚠️ *यह केवल सूचनात्मक उद्देश्यों के लिए है और चिकित्सा सलाह नहीं है। "
    "उचित निदान और उपचार के लिए कृपया किसी योग्य चिकित्सक से परामर्श करें।*"
)

# Emergency response — bypasses LLM entirely
EMERGENCY_RESPONSE = (
    "🚨 **This sounds like a medical emergency.**\n\n"
    "Please take immediate action:\n"
    "• **Call 112** (India Emergency) or **108** (Ambulance)\n"
    "• Go to the nearest hospital emergency department\n"
    "• If someone is with you, ask them to help immediately\n\n"
    "I am an AI assistant and cannot provide emergency medical care. "
    "Please seek professional help right away."
)

EMERGENCY_RESPONSE_HI = (
    "🚨 **यह एक चिकित्सा आपातकाल लगता है।**\n\n"
    "कृपया तुरंत कार्रवाई करें:\n"
    "• **112 पर कॉल करें** (भारत आपातकालीन) या **108** (एम्बुलेंस)\n"
    "• निकटतम अस्पताल के आपातकालीन विभाग में जाएं\n"
    "• अगर कोई आपके साथ है, तो तुरंत मदद मांगें\n\n"
    "मैं एक AI सहायक हूं और आपातकालीन चिकित्सा देखभाल प्रदान नहीं कर सकता।"
)


SAFETY_CHECK_PROMPT = """Check if this response contains any of these issues:
1. Specific dosage recommendation without a doctor's prescription
2. Diagnosis of a serious condition without recommending professional consultation
3. Advice to stop or change prescribed medication
4. Claims of being able to cure or treat a condition

User question: {question}
Response to check: {response}

Reply with ONLY "SAFE" or "UNSAFE: <reason>". Nothing else."""


def check_emergency(text: str) -> bool:
    """Check if user message contains emergency keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)


def detect_language(text: str) -> str:
    """Simple Hindi vs English detection based on Unicode ranges."""
    hindi_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    if hindi_chars > len(text) * 0.15:
        return "hi"
    return "en"


def get_disclaimer(lang: str = "en") -> str:
    """Get disclaimer in the appropriate language."""
    return DISCLAIMER_HI if lang == "hi" else DISCLAIMER


def get_emergency_response(lang: str = "en") -> str:
    """Get emergency response in the appropriate language."""
    return EMERGENCY_RESPONSE_HI if lang == "hi" else EMERGENCY_RESPONSE
