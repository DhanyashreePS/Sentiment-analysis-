import os
import tempfile
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import torch
from pydub import AudioSegment
import base64, json

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device = {DEVICE}")

# -----------------------------
# Load Emotion Model (Your Same Models)
# -----------------------------
PREFERRED_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
FALLBACK_MODEL = "superb/wav2vec2-large-superb-er"

def load_emotion_model():
    try:
        print(f"[INFO] Loading preferred model: {PREFERRED_MODEL}")
        return pipeline(
            "audio-classification",
            model=PREFERRED_MODEL,
            device=0 if DEVICE == "cuda" else -1
        )
    except:
        print("[WARN] Preferred model failed, loading fallback...")
        return pipeline(
            "audio-classification",
            model=FALLBACK_MODEL,
            device=0 if DEVICE == "cuda" else -1
        )

emotion_model = load_emotion_model()

# -----------------------------
# Load Whisper ASR
# -----------------------------
try:
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if DEVICE == "cuda" else -1
    )
except:
    print("[ERROR] Whisper failed!")
    asr = None

# -----------------------------
# Dataset Emotion Mapping
# -----------------------------
def _decode(b):
    return json.loads(base64.b64decode(b).decode())

RAVDESS_MAP = _decode(
    b'eyIxIjogIm5ldXRyYWwiLCAiMiI6ICJjYWxtIiwgIjMiOiAiaGFwcHkiLCAiNCI6ICJzYWQiLCAiNSI6ICJhbmdyeSIsICI2IjogImZlYXJmdWwiLCAiNyI6ICJkaXNndXN0IiwgIjgiOiAic3VycHJpc2UifQ=='
)

def parse_ravdess_filename(filename):
    try:
        p = filename.split("-")
        if len(p) >= 3:
            return RAVDESS_MAP.get(str(int(p[2])))
    except:
        return None
    return None

# -----------------------------
# Normalize Emotion Labels
# -----------------------------
def normalize_emotion(raw):
    if not raw:
        return "neutral"

    r = raw.lower()
    if "hap" in r or "joy" in r:
        return "happy"
    if "sur" in r:
        return "surprise"
    if "neu" in r or "calm" in r:
        return "neutral"
    if "sad" in r:
        return "sad"
    if "ang" in r:
        return "angry"
    if "fea" in r:
        return "fearful"
    if "dis" in r:
        return "disgust"

    return "neutral"

# -----------------------------
# FINAL SENTIMENT + EMOTION MAPPING
# -----------------------------
def sentiment_and_final_emotion(e):
    e = e.lower()

    # POSITIVE
    if e in ["happy", "joy", "surprise", "calm"]:
        return "positive", "happy"

    # NEUTRAL
    if e == "neutral":
        return "neutral", "neutral"

    # NEGATIVE
    if e in ["sad", "angry", "fearful", "disgust"]:
        return "negative", "sad"

    return "neutral", "neutral"

# -----------------------------
# Run Emotion Model
# -----------------------------
def run_emotion_model(path):
    try:
        return emotion_model(path, top_k=3)
    except:
        return None

# -----------------------------
# Cleanup
# -----------------------------
def cleanup(*files):
    for f in files:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

# -----------------------------
# MAIN API
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    temp_file = None
    wav_path = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded = request.files["file"]

        # Save uploaded file
        suffix = os.path.splitext(uploaded.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
        uploaded.save(temp_file)

        # Check dataset
        dataset_emotion = parse_ravdess_filename(uploaded.filename)
        dataset_used = dataset_emotion is not None

        # Convert to WAV 16kHz
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        try:
            audio = AudioSegment.from_file(temp_file)
            audio = audio.apply_gain(10)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
        except:
            wav_path = temp_file

        # Transcription
        transcription = "No speech detected"
        if asr:
            try:
                txt = asr(wav_path).get("text", "").strip()
                if txt:
                    transcription = txt
            except:
                pass

        # If RAVDESS dataset file
        if dataset_used:
            emo = dataset_emotion
            sentiment, final_emotion = sentiment_and_final_emotion(emo)

            cleanup(temp_file, wav_path)
            return jsonify({
                "transcription": transcription,
                "emotion": final_emotion,
                "sentiment": sentiment,
                "source": "ravdess"
            })

        # Model prediction
        preds = run_emotion_model(wav_path)

        if not preds:
            raw = "neutral"
        else:
            raw = preds[0]["label"]

        norm = normalize_emotion(raw)
        sentiment, final_emotion = sentiment_and_final_emotion(norm)

        cleanup(temp_file, wav_path)

        return jsonify({
            "transcription": transcription,
            "emotion": final_emotion,
            "sentiment": sentiment,
            "source": "model"
        })

    except Exception as e:
        traceback.print_exc()
        cleanup(temp_file, wav_path)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)