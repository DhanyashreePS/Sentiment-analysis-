# 🎧 Audio Sentiment & Emotion Analysis System

## 📖 Overview

The Audio Sentiment & Emotion Analysis System is an AI-powered web application that analyzes emotions from speech audio. Users can upload a speech audio file, and the system predicts the speaker's emotion using a deep learning model. It also converts speech into text and displays the transcription along with the detected emotion.

This project combines Machine Learning, Speech Processing, and Full-Stack Web Development to provide an interactive and intelligent emotion recognition system.

---

## 🚀 Features

- Upload speech audio files (.wav)
- Automatic emotion detection
- Speech-to-text transcription
- Displays predicted emotion and sentiment
- Interactive React-based user interface
- RESTful API built with Flask
- Deep learning-based emotion classification

---

## 🛠️ Technologies Used

### Frontend
- React.js
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning
- TensorFlow / Keras
- Scikit-learn
- NumPy

### Audio Processing
- Librosa
- SpeechRecognition

### Dataset
- Toronto Emotional Speech Set (TESS)

---

## 📂 Project Structure

```
sentiment-project/
│
├── backend/
│   ├── dataset/
│   │   └── AudioWAV/
│   ├── app.py
│   ├── train.py
│   ├── emotion_model.h5
│   ├── scaler.pkl
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── assets/
│   ├── package.json
│   └── node_modules/
│
└── README.md
```

---

## ⚙️ System Workflow

1. User uploads a speech audio (.wav) file.
2. Flask backend receives the uploaded audio.
3. Librosa extracts MFCC features from the audio.
4. The trained TensorFlow model predicts the emotion.
5. SpeechRecognition converts speech into text.
6. Backend sends the emotion, sentiment, and transcription to the frontend.
7. React displays the prediction results.

---

## 🧠 Machine Learning Workflow

### Dataset
The model is trained using the Toronto Emotional Speech Set (TESS) dataset.

### Feature Extraction
The audio is processed using Mel-Frequency Cepstral Coefficients (MFCC), which convert speech into meaningful numerical features.

### Model
A Deep Neural Network (DNN) is trained using TensorFlow/Keras to classify emotions from extracted features.

### Predicted Emotions

- Happy 😊
- Sad 😢
- Angry 😡
- Fear 😨
- Disgust 🤢
- Neutral 😐

---

## ▶️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/audio-sentiment-analysis.git
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment (Windows)

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow flask flask-cors librosa numpy scikit-learn SpeechRecognition
```

---

## ▶️ Train the Model

```bash
python train.py
```

This generates the trained model:

```
emotion_model.h5
```

---

## ▶️ Run the Backend

```bash
python app.py
```

Backend URL:

```
http://127.0.0.1:5000
```

---

## ▶️ Run the Frontend

```bash
cd frontend
npm install
npm start
```

Frontend URL:

```
http://localhost:3000
```

---

## 📥 Input

- Speech Audio (.wav)

## 📤 Output

- Speech Transcription
- Predicted Emotion
- Predicted Sentiment

---

## 🌍 Real-World Applications

- Customer Support Call Analysis
- Mental Health Monitoring
- Virtual Voice Assistants
- Online Interview Analysis
- Human-Computer Interaction
- E-learning Platforms
- Smart Call Centers

---

## ⚠️ Challenges Faced

- Handling different audio formats
- Improving prediction accuracy
- Audio feature extraction
- Frontend and backend integration
- Speech-to-text implementation

---

## 🔮 Future Enhancements

- Real-time microphone support
- CNN/LSTM-based emotion recognition
- Noise reduction techniques
- Multilingual speech recognition
- Cloud deployment
- Transformer-based speech models
- Advanced sentiment analysis using NLP

---

## 📌 Conclusion

The Audio Sentiment & Emotion Analysis System demonstrates how Artificial Intelligence and Speech Processing can be combined to recognize emotions from human speech. By integrating audio feature extraction, deep learning, speech-to-text, and a responsive web interface, the system provides an efficient solution for emotion recognition with applications in healthcare, customer service, education, and intelligent virtual assistants.

---

## 👩‍💻 Author

**DHANYASHREE P S**

---

## 📄 License

This project is developed for academic and educational purposes. It can be modified and extended for learning and research.
