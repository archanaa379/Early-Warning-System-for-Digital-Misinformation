# ============================================================
# Early Warning System for Digital Misinformation-Driven Panic
# Flask Backend — app.py
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import math
from collections import Counter

app = Flask(__name__)
CORS(app)  # Allow frontend (HTML) to talk to this backend

# ── Stopwords (mini version — install nltk for full list) ──
STOPWORDS = {
    'i','me','my','we','our','you','your','he','she','it','they','them',
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might',
    'a','an','the','and','or','but','if','in','on','at','to','for',
    'of','with','by','from','as','this','that','these','those','so',
    'not','no','nor','very','just','also','about','up','out','then'
}

# ── Crisis / Panic Keywords with weights ──
PANIC_KEYWORDS = {
    'panic': 0.9, 'collapse': 0.85, 'shortage': 0.8, 'outbreak': 0.85,
    'lockdown': 0.7, 'disaster': 0.8, 'riot': 0.85, 'failure': 0.7,
    'danger': 0.75, 'emergency': 0.75, 'crisis': 0.8, 'fear': 0.65,
    'attack': 0.8, 'kill': 0.9, 'dead': 0.85, 'dying': 0.9,
    'flood': 0.7, 'earthquake': 0.75, 'explosion': 0.85,
    'no cure': 0.95, 'no treatment': 0.95, 'spreading': 0.7,
    'death toll': 0.9, 'mass casualty': 0.95, 'imminent': 0.8,
    'run out': 0.75, 'contaminated': 0.8, 'poisoned': 0.85,
    'evacuate': 0.75, 'trapped': 0.8, 'helpless': 0.7
}

MISINFO_KEYWORDS = {
    'they are hiding': 0.95, 'cover up': 0.9, 'cover-up': 0.9,
    'secret': 0.6, 'government is lying': 0.95, 'fake': 0.7,
    'hoax': 0.85, 'they don\'t want you': 0.9, 'miracle': 0.75,
    'cure all': 0.85, '100%': 0.55, 'doctors don\'t want': 0.9,
    'no one is telling': 0.9, 'suppressed': 0.85, 'silenced': 0.8,
    'mainstream media': 0.65, 'they won\'t tell': 0.9,
    'share before deleted': 1.0, 'share immediately': 0.8,
    'wake up': 0.7, 'sheeple': 0.9, 'big pharma': 0.75,
    'deep state': 0.9, 'plandemic': 0.95, 'fake news': 0.6,
    'unverified': 0.5, 'rumor': 0.55
}

NEGATIVE_WORDS = {
    'horrible': 0.7, 'terrible': 0.65, 'awful': 0.6, 'devastating': 0.8,
    'catastrophic': 0.85, 'alarming': 0.75, 'threatening': 0.8,
    'dangerous': 0.75, 'deadly': 0.85, 'fatal': 0.85, 'severe': 0.7,
    'tragic': 0.75, 'horrific': 0.85, 'nightmare': 0.75, 'worse': 0.55,
    'worst': 0.65, 'unbearable': 0.7, 'shocking': 0.65, 'outrageous': 0.7
}

CREDIBILITY_BOOSTERS = {
    'according to': 0.3, 'study shows': 0.35, 'research indicates': 0.35,
    'official': 0.3, 'confirmed': 0.3, 'reported by': 0.25,
    'data shows': 0.35, 'survey': 0.25, 'statistics': 0.3,
    'peer reviewed': 0.4, 'published': 0.3, 'ministry': 0.35,
    'government announced': 0.35, 'scientist': 0.3, 'doctor': 0.25,
    'hospital': 0.2, 'university': 0.3, 'percent': 0.2
}


# ── 1. Text Preprocessing ──────────────────────────────────
def preprocess(text):
    """Clean and tokenize text (mimics NLTK pipeline)."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)             # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()           # collapse whitespace
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return tokens, text  # return tokens + cleaned string


# ── 2. Keyword Scorer ─────────────────────────────────────
def keyword_score(cleaned_text, keyword_dict):
    """
    Score text against a weighted keyword dictionary.
    Uses phrase matching on cleaned text.
    """
    total_weight = 0.0
    matched = []
    for phrase, weight in keyword_dict.items():
        if phrase in cleaned_text:
            total_weight += weight
            matched.append(phrase)
    # Normalize: cap at 1.0 using sigmoid-like compression
    normalized = 1 - math.exp(-total_weight * 0.8)
    return round(min(normalized, 0.99), 3), matched


# ── 3. TF-IDF Feature Simulation ──────────────────────────
def tfidf_features(tokens):
    """
    Simple TF computation (IDF would need a corpus — 
    this simulates what sklearn TfidfVectorizer does).
    Returns top weighted tokens.
    """
    freq = Counter(tokens)
    total = len(tokens) if tokens else 1
    tf = {word: count / total for word, count in freq.items()}
    # Sort by frequency as a TF-IDF proxy
    top_tokens = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
    return [t[0] for t in top_tokens]


# ── 4. Sentiment Analysis ─────────────────────────────────
def sentiment_score(cleaned_text, tokens):
    """
    Rule-based polarity scoring (mimics TextBlob sentiment).
    Returns a score from 0 (positive) to 1 (very negative).
    """
    neg_score, matched = keyword_score(cleaned_text, NEGATIVE_WORDS)
    # Also check for ALL CAPS words (signals alarm)
    original_words = cleaned_text.upper().split()
    caps_ratio = sum(1 for w in original_words if len(w) > 3) / max(len(original_words), 1)
    combined = min(neg_score + caps_ratio * 0.1, 0.99)
    return round(combined, 3)


# ── 5. Credibility Check ──────────────────────────────────
def credibility_score(cleaned_text):
    """Check for credibility boosters (official sources, data citations)."""
    score = 0.0
    found = []
    for phrase, weight in CREDIBILITY_BOOSTERS.items():
        if phrase in cleaned_text:
            score += weight
            found.append(phrase)
    # Higher credibility → lower misinfo likelihood
    credibility = round(min(score, 1.0), 3)
    return credibility, found


# ── 6. Final Classification ───────────────────────────────
def classify(misinfo_score, panic_score, sentiment, credibility):
    """
    Simulate Random Forest output:
    0 = Normal, 1 = Misinformation, 2 = Panic-Inducing Misinformation
    """
    # Reduce scores if credibility boosters found
    adjusted_misinfo = max(misinfo_score - credibility * 0.4, 0)
    adjusted_panic   = max(panic_score   - credibility * 0.2, 0)

    if adjusted_misinfo > 0.55 or adjusted_panic > 0.60:
        label = 2
        category = "PANIC-INDUCING MISINFORMATION"
        risk = "HIGH"
        action = "Immediate review and public clarification recommended. Alert dispatched."
    elif adjusted_misinfo > 0.28 or adjusted_panic > 0.30:
        label = 1
        category = "POSSIBLE MISINFORMATION"
        risk = "MEDIUM"
        action = "Cross-verify with trusted sources before sharing."
    else:
        label = 0
        category = "LIKELY LEGITIMATE"
        risk = "LOW"
        action = "No immediate action required. Continue monitoring."

    return {
        "label": label,
        "category": category,
        "risk_level": risk,
        "action": action,
        "adjusted_misinfo": round(adjusted_misinfo, 3),
        "adjusted_panic": round(adjusted_panic, 3)
    }


# ══════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/')
def home():
    return jsonify({
        "system": "Early Warning System for Digital Misinformation-Driven Panic",
        "status": "online",
        "version": "1.0",
        "endpoints": ["/analyze", "/health", "/keywords"]
    })


@app.route('/health')
def health():
    return jsonify({"status": "online", "pipeline": "active"})


@app.route('/keywords')
def keywords():
    """Return all crisis keywords for the frontend to display."""
    return jsonify({
        "panic_keywords":  list(PANIC_KEYWORDS.keys()),
        "misinfo_keywords": list(MISINFO_KEYWORDS.keys()),
        "negative_words":  list(NEGATIVE_WORDS.keys())
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    POST body: { "text": "...", "category": "health", "threshold": 0.65 }
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text      = data.get('text', '').strip()
    category  = data.get('category', 'general')
    threshold = float(data.get('threshold', 0.65))

    if len(text) < 5:
        return jsonify({"error": "Text too short to analyze"}), 400

    # ── Run Pipeline ──
    tokens, cleaned = preprocess(text)

    misinfo_score, misinfo_matches   = keyword_score(cleaned, MISINFO_KEYWORDS)
    panic_score,   panic_matches     = keyword_score(cleaned, PANIC_KEYWORDS)
    sentiment                        = sentiment_score(cleaned, tokens)
    credibility, cred_matches        = credibility_score(cleaned)
    top_features                     = tfidf_features(tokens)
    result                           = classify(misinfo_score, panic_score, sentiment, credibility)

    # ── Alert threshold check ──
    alert_triggered = (
        result["adjusted_misinfo"] >= threshold or
        result["adjusted_panic"] >= threshold
    )

    return jsonify({
        "input": {
            "text_length": len(text),
            "word_count": len(tokens),
            "category": category,
            "threshold": threshold
        },
        "pipeline": {
            "step1_preprocessing": {
                "tokens_extracted": len(tokens),
                "top_tokens": top_features
            },
            "step2_misinfo_score": misinfo_score,
            "step3_panic_score": panic_score,
            "step4_sentiment_score": sentiment,
            "step5_credibility_score": credibility
        },
        "keywords_found": {
            "misinfo_triggers": misinfo_matches,
            "panic_triggers": panic_matches,
            "credibility_signals": cred_matches
        },
        "classification": result,
        "alert_triggered": alert_triggered,
        "recommendation": result["action"]
    })


# ── Run ──
if __name__ == '__main__':
    print("\n🚀 Early Warning System Backend — Running on http://localhost:5000")
    print("📡 Endpoints: /  |  /analyze  |  /health  |  /keywords\n")
    app.run(debug=True, port=5000)
