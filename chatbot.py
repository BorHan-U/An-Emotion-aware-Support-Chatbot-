import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk  # imported to show NLTK usage (no downloads needed)
import time
from transformers import MarianMTModel, MarianTokenizer # added for german response



#******for ml evaluator---------
from transformers import pipeline

# === Emotion Confidence Classifier (ML-based) ===
print("Loading emotion classifier...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
print("Emotion classifier loaded.")
#****-------------------

#added for german response
print("Loading translation models...")
DE_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
EN_DE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"

de_en_tokenizer = MarianTokenizer.from_pretrained(DE_EN_MODEL_NAME)
de_en_model = MarianMTModel.from_pretrained(DE_EN_MODEL_NAME)

en_de_tokenizer = MarianTokenizer.from_pretrained(EN_DE_MODEL_NAME)
en_de_model = MarianMTModel.from_pretrained(EN_DE_MODEL_NAME)

print("Translation models loaded.")
#added for german response







# -------------------------------------------------------
# Flask app
# -------------------------------------------------------

app = Flask(__name__)

# -------------------------------------------------------
# Load local Gemma-2B-Instruct model
# -------------------------------------------------------

MODEL_NAME = "google/gemma-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {MODEL_NAME} on {DEVICE}... (first run can take a few minutes)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# For CPU we just use float32 (slower but simple & safe)
if DEVICE == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.to(DEVICE)

model.eval()

print("Model loaded.")


def generate_with_gemma(prompt: str, max_new_tokens: int = 160) -> str:
    """
    Generate a continuation from Gemma using the chat template.
    """
    # Wrap the prompt into a chat-style conversation
    chat = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Gemma has a built-in chat template
    rendered = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(rendered, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode the *new* tokens (not the prompt)
    generated_ids = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


# -------------------------------------------------------
# Emotion detection (simple keyword-based)
# -------------------------------------------------------

EMOTION_KEYWORDS = {
    "sadness": [
        "sad", "down", "depressed", "unhappy", "miserable",
        "hopeless", "empty", "cry", "crying", "heartbroken",
    ],
    "anxiety": [
        "anxious", "worried", "nervous", "panic", "panicking",
        "stressed", "overwhelmed", "afraid", "scared",
    ],
    "anger": [
        "angry", "mad", "furious", "irritated", "annoyed",
        "rage", "pissed", "frustrated",
    ],
    "loneliness": [
        "lonely", "alone", "isolated", "abandoned", "ignored",
    ],
    "positive": [
        "happy", "good", "great", "excited", "proud",
        "hopeful", "relieved", "calm", "grateful", "thankful",
    ],
}


#*******ML confident score-------
def detect_emotion_with_confidence(text: str):
    """
    Returns:
        dominant_emotion (str)
        confidence (float between 0 and 1)
        all_scores (dict of all emotion probabilities)
    """

    results = emotion_classifier(text)[0]  # list of emotion scores

    scores = {item["label"].lower(): item["score"] for item in results}

    dominant_emotion = max(scores, key=scores.get)
    confidence = float(scores[dominant_emotion])

    return dominant_emotion, confidence, scores


#added for german response
def translate_text(text: str, direction: str) -> str:
    """
    direction: 'de-en' or 'en-de'
    """
    if not text.strip():
        return text

    if direction == "de-en":
        tokenizer = de_en_tokenizer
        model = de_en_model
    elif direction == "en-de":
        tokenizer = en_de_tokenizer
        model = en_de_model
    else:
        return text  # no translation

    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=256)
    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return translated[0] if translated else text
#added for german response

#***---------------
'''
def detect_emotion(text: str) -> str:
    """
    Very simple keyword-based emotion detection.
    Returns one of: sadness, anxiety, anger, loneliness, positive, neutral
    """
    text_lower = text.lower()
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}

    for emotion, words in EMOTION_KEYWORDS.items():
        for w in words:
            if w in text_lower:
                scores[emotion] += 1

    best_emotion = max(scores, key=scores.get)
    if scores[best_emotion] == 0:
        return "neutral"
    return best_emotion
'''

# -------------------------------------------------------
# Crisis / safety detection
# -------------------------------------------------------

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm",
    "hurt myself", "can't go on", "dont want to live", "don't want to live",
]


def crisis_check(message: str) -> bool:
    text_lower = message.lower()
    return any(k in text_lower for k in CRISIS_KEYWORDS)


def supportive_crisis_message() -> str:
    return (
        "I'm really sorry you're feeling this much pain.\n\n"
        "I'm not a professional and I can't keep you safe in an emergency, "
        "but you **deserve immediate human support**:\n\n"
        "• If you are in immediate danger, please contact your local emergency services right away.\n"
        "• You can also reach out to a trusted friend, family member, or local mental health professional.\n\n"
        "You don't have to go through this alone."
    )


# -------------------------------------------------------
# Response generation (Gemma + emotion)
# -------------------------------------------------------

def generate_empathetic_response(user_message: str, emotion: str, confidence: float) -> str:
    """
    Confidence-aware response generation:
    - If confidence is LOW (< 0.4): Ask for clarification
    - If confidence is NORMAL: Generate emotion-aware empathetic reply
    """

    # ✅ LOW CONFIDENCE MODE — Be cautious
    if confidence < 0.4:
        return (
            "I might be misunderstanding how you're feeling right now, "
            "and I don’t want to assume anything. "
            "If you feel comfortable, could you tell me a bit more about what’s going on for you?"
        )

    # ✅ NORMAL MODE — Use Gemma for emotion-aware support
    prompt = (
        "You are an empathetic, non-clinical mental health support companion.\n"
        "Use 2-4 warm, supportive sentences.\n"
        "Do NOT diagnose.\n"
        "Do NOT give medical advice.\n\n"
        f"Detected emotion: {emotion}\n"
        f"User message: {user_message}\n\n"
        "Reply:"
    )

    try:
        reply = generate_with_gemma(prompt)

        if not reply.strip():
            raise ValueError("Empty model output")

        if len(reply) > 500:
            reply = reply[:500]

        return reply

    except Exception as e:
        print("Generation error:", e)
        return (
            "Thank you for sharing that with me. I’m here to listen. "
            "You can tell me more if you’d like."
        )


'''
def generate_empathetic_response(user_message: str, emotion: str) -> str:
    """
    Use Gemma-2B-Instruct to generate a short, empathetic response.
    100% local, no OpenAI.
    """

    prompt = (
        "You are an empathetic, non-clinical mental health support companion.\n"
        "Guidelines:\n"
        "- Use a warm, validating tone.\n"
        "- 2-5 sentences.\n"
        "- Do NOT diagnose.\n"
        "- Do NOT give medical or medication advice.\n"
        "- Encourage reaching out to trusted people or professionals if things are very hard.\n\n"
        f"Detected emotion: {emotion}\n\n"
        f"User message: {user_message}\n\n"
        "Write your reply to the user:"
    )

    reply = generate_with_gemma(prompt)

    # Basic safety net
    if not reply:
        reply = (
            "Thank you for sharing that with me. I'm here to listen. "
            "If you'd like, you can tell me a bit more about how this feels for you."
        )

    # Hard cut-off in case the model rambles
    if len(reply) > 800:
        reply = reply[:800]

    return reply
'''

# -------------------------------------------------------
# Flask routes
# -------------------------------------------------------

@app.route("/")
def index():
    # Your existing templates/index.html will still work
    return render_template("index.html")


#added for german response

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_lang = data.get("lang", "en")  # 'en' or 'de'

    start_time = time.time()

    if not user_message:
        end_time = time.time()
        return jsonify({
            "reply": "I’m here whenever you feel ready to share something.",
            "emotion": "neutral",
            "confidence": 0.0,
            "crisis": False,
            "response_time": round(end_time - start_time, 3)
        })

    # If message is German, translate to English for internal processing
    if user_lang == "de":
        processed_text = translate_text(user_message, "de-en")
    else:
        processed_text = user_message

    # 1. Crisis check (run on English text)
    if crisis_check(processed_text):
        # Crisis response is currently in English; you may translate to German
        crisis_reply = supportive_crisis_message()
        if user_lang == "de":
            crisis_reply = translate_text(crisis_reply, "en-de")

        end_time = time.time()
        return jsonify({
            "reply": crisis_reply,
            "emotion": "crisis",
            "confidence": 1.0,
            "crisis": True,
            "response_time": round(end_time - start_time, 3)
        })

    # 2. Emotion detection on English text
    emotion, confidence, all_scores = detect_emotion_with_confidence(processed_text)

    # 3. Generate response (Gemma) using English text + emotion + confidence
    bot_reply_en = generate_empathetic_response(processed_text, emotion, confidence)

    # If user language is German, translate bot reply back to German
    if user_lang == "de":
        bot_reply = translate_text(bot_reply_en, "en-de")
    else:
        bot_reply = bot_reply_en

    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    return jsonify({
        "reply": bot_reply,
        "emotion": emotion,
        "confidence": float(confidence),
        "crisis": False,
        "response_time": response_time
    })

#added for german response



'''
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    start_time = time.time()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "reply": "I'm here whenever you feel ready to share something.",
            "emotion": "neutral",
            "crisis": False,
        })

    # 1. Crisis check
    if crisis_check(user_message):
        return jsonify({
            "reply": supportive_crisis_message(),
            "emotion": "crisis",
            "crisis": True,
        })

    # 2. Emotion detection
    # emotion = detect_emotion(user_message)

    #*************ml confident-----------
    emotion, confidence, all_scores = detect_emotion_with_confidence(user_message)


    # 3. Generate response with Gemma
     #*************ml confident-----------
    bot_reply = generate_empathetic_response(user_message, emotion, confidence)
    # reply = generate_empathetic_response(user_message, emotion)

    #*************ml confident-----------
    end_time = time.time()
    response_time = round(end_time - start_time, 3)
    return jsonify({
        "reply": bot_reply,
        "emotion": emotion,
        "confidence": round(confidence, 3),
        "response_time": response_time
    })
    '''
    
    #return jsonify({
        #"reply": reply,
        #"emotion": emotion,
        #"crisis": False,
    #})


if __name__ == "__main__":
    app.run(debug=True)

























'''
import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import nltk  # imported, but we'll avoid features that require downloaded corpora


#----****------------

from transformers import pipeline, set_seed

# Load a small GPT-2 model for offline generation
local_generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=60,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256  # suppress padding warnings
)

set_seed(42)


"""from transformers import pipeline

local_generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=80,
    temperature=0.7
)"""
#-------*****---------



# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# Read API key from environment variable (recommended)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)

# -------------------------------------------------------
# Simple Rule-Based Emotion Detection (No external models)
# -------------------------------------------------------

EMOTION_KEYWORDS = {
    "sadness": [
        "sad", "down", "depressed", "unhappy", "tearful", "miserable",
        "hopeless", "worthless", "empty", "cry", "crying"
    ],
    "anxiety": [
        "anxious", "worried", "nervous", "panic", "panicking", "stressed",
        "overwhelmed", "afraid", "scared"
    ],
    "anger": [
        "angry", "mad", "furious", "irritated", "annoyed", "rage",
        "pissed", "frustrated"
    ],
    "loneliness": [
        "lonely", "alone", "isolated", "abandoned", "ignored"
    ],
    "positive": [
        "happy", "good", "great", "excited", "proud", "hopeful",
        "relieved", "calm"
    ],
}


def detect_emotion(text: str) -> str:
    """
    Very simple keyword-based emotion detection.
    Returns one of: sadness, anxiety, anger, loneliness, positive, neutral
    """
    text_lower = text.lower()
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS.keys()}

    for emotion, words in EMOTION_KEYWORDS.items():
        for w in words:
            if w in text_lower:
                scores[emotion] += 1

    # find emotion with max score
    best_emotion = max(scores, key=scores.get)
    if scores[best_emotion] == 0:
        return "neutral"
    return best_emotion


# -------------------------------------------------------
# Crisis / Safety Detection
# -------------------------------------------------------

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm",
    "hurt myself", "can't go on", "don't want to live"
]


def crisis_check(message: str) -> bool:
    text_lower = message.lower()
    return any(k in text_lower for k in CRISIS_KEYWORDS)


def supportive_crisis_message() -> str:
    return (
        "I'm really sorry that you're feeling this much pain. 💛\n\n"
        "I’m not a professional and I can’t keep you safe in an emergency, "
        "but you **deserve immediate human support**:\n\n"
        "• If you are in immediate danger, **please contact your local emergency services right away**.\n"
        "• You can also reach out to a trusted friend, family member, or local mental health professional.\n"
        "• Many countries have crisis hotlines you can call or chat with online.\n\n"
        "You don’t have to go through this alone."
    )




#-------------****------
def generate_empathetic_response(user_message: str, emotion: str) -> str:
    # ------------------------------
    # Free, local model fallback
    # ------------------------------
    if client is None:
        prompt = (
            f"You are an empathetic mental health companion.\n"
            f"User emotion: {emotion}\n"
            f"Respond warmly, supportively, and briefly.\n"
            f"User: {user_message}\n"
            f"Bot:"
        )

        raw_output = local_generator(prompt)[0]["generated_text"]

        # Extract only what comes after "Bot:"
        if "Bot:" in raw_output:
            cleaned = raw_output.split("Bot:", 1)[-1]
        else:
            cleaned = raw_output

        # Remove repeated patterns
        cleaned = cleaned.strip()
        lines = cleaned.splitlines()
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        cleaned = " ".join(unique_lines)

        # Limit extremely long outputs
        cleaned = cleaned[:350]

        return cleaned
    """if client is None:
        prompt = (
            f"The user feels {emotion}. Respond in a warm, empathetic, supportive way. "
            f"Do not diagnose. Keep it under 60 words.\n"
            f"User: {user_message}\nBot:"
        )

        output = local_generator(prompt)[0]["generated_text"]

        # clean up the output after "Bot:"
        if "Bot:" in output:
            output = output.split("Bot:", 1)[-1]

        return output.strip()"""

    # ------------------------------
    # OpenAI version (unchanged)
    # ------------------------------
    prompt = f"""
You are an empathetic, non-clinical mental health support chatbot.
The user feels: {emotion}.
Respond in a warm, validating, supportive way under 120 words.
Avoid medical advice and diagnosis.

User message:
\"\"\"{user_message}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a kind, supportive companion."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("OpenAI error:", e)
        return (
            "I’m still here with you. I’m having trouble accessing my AI model right now, "
            "but I can listen. What’s on your mind?"
        )

"""
# -------------------------------------------------------
# Response Generation (with safe fallback if no OpenAI key)
# -------------------------------------------------------
def generate_empathetic_response(user_message: str, emotion: str) -> str:
    """
    If OPENAI_API_KEY is set, use OpenAI for richer responses.
    Otherwise, return a simple rule-based empathetic message.
    """

    # Fallback if no API key
    if client is None:
        base = "Thank you for sharing that with me. "
        if emotion == "sadness":
            return (
                base +
                "It sounds like you're feeling really low right now. "
                "It's okay to feel this way; your feelings are valid. "
                "If you want, you can tell me more about what's been weighing on you."
            )
        elif emotion == "anxiety":
            return (
                base +
                "It seems like you're feeling very anxious or overwhelmed. "
                "Try to take a slow, deep breath. "
                "If you'd like, we can talk through what's making you feel this way."
            )
        elif emotion == "anger":
            return (
                base +
                "I can sense a lot of frustration or anger. "
                "Those feelings are understandable. "
                "Would you like to talk about what triggered them?"
            )
        elif emotion == "loneliness":
            return (
                base +
                "Feeling alone can be really hard. "
                "Even though I'm just a bot, I'm here to listen. "
                "You can tell me what’s making you feel this way."
            )
        elif emotion == "positive":
            return (
                base +
                "It sounds like you're feeling relatively positive. "
                "That's wonderful. If you’d like, we can reflect on what’s going well for you right now."
            )
        else:
            return (
                base +
                "I'm here with you. "
                "How are things feeling for you in this moment?"
            )

    # If OpenAI client is available, use it
    prompt = f"""
You are an empathetic, non-clinical mental health support chatbot.
The user feels: {emotion}.
Your job is to respond in a warm, validating, and supportive way.
You MUST:
- Avoid giving medical advice.
- Avoid diagnosing.
- Encourage reaching out to real people or professionals if things are very hard.
Keep your answer under 120 words.

User message:
\"\"\"{user_message}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a kind, supportive companion."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Safe fallback if API fails
        print("OpenAI error:", e)
        return (
            "I'm really glad you shared that with me. "
            "I’m having trouble accessing my language model right now, "
            "but I’m still here with you. How would you describe what you’re feeling in a few words?"
        )"""


# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "reply": "I’m here whenever you feel ready to share something.",
            "emotion": "neutral",
            "crisis": False,
        })

    # Crisis check first
    if crisis_check(user_message):
        return jsonify({
            "reply": supportive_crisis_message(),
            "emotion": "crisis",
            "crisis": True,
        })

    # Emotion detection
    emotion = detect_emotion(user_message)

    # Response generation
    reply = generate_empathetic_response(user_message, emotion)

    return jsonify({
        "reply": reply,
        "emotion": emotion,
        "crisis": False,
    })


if __name__ == "__main__":
    # For local development (VS Code, etc.)
    app.run(debug=True)
'''
