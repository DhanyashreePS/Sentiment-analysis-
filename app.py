# app.py
import os
import numpy as np
import librosa
import tensorflow as tf
import speech_recognition as sr

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("emotion_model.h5")

emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad"
}

def extract_features(file_path):
    try:
        audio, sr_rate = librosa.load(file_path, duration=3, offset=0.5)
        if len(audio) == 0:
            return None
        mfcc = librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Feature error:", e)
        return None
    
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text

    except Exception as e:
        print("Transcription error:", e)
        return "Could not understand audio"

@app.route("/")
def home():
    return "Backend Running ✅"

@app.route("/predict", methods=["POST"])
@app.route("/analyze", methods=["POST"])
def analyze():
    print("🔥 Request received")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    print("📁 File:", file.filename)

    temp_path = "temp.wav"
    file.save(temp_path)

    try:
        transcription = transcribe_audio(temp_path)
        print("🗣 Transcription:", transcription)
        
        features = extract_features(temp_path)

        if features is None:
            return jsonify({"error": "Invalid audio file"})

        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_class = int(np.argmax(prediction))

        emotion = emotion_labels.get(predicted_class, "unknown")

        print("🎯 Emotion:", emotion)

        return jsonify({
            "transcription": transcription,
            "sentiment": emotion,
            "emotion": emotion
        })

    except Exception as e:
        print("💥 ERROR:", e)
        return jsonify({"error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
if __name__ == "__main__":
    app.run(debug=True)
