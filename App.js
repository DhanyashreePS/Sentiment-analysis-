import React, { useState, useEffect } from "react";
import "./App.css";
import emotionRecognition from "./assets/emotion-recognition.png";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("No file chosen");
  const [audioURL, setAudioURL] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const emotionEmoji = {
    neutral: "😐",
    calm: "😌",
    happy: "😊",
    joy: "😄",
    surprise: "😮",
    angry: "😡",
    disgust: "🤢",
    fearful: "😱",
    sad: "😢"
  };

  useEffect(() => {
    return () => {
      if (audioURL) URL.revokeObjectURL(audioURL);
    };
  }, [audioURL]);

  const allowedExtensions = ["wav", "mp3", "m4a"];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const ext = selectedFile.name.split(".").pop().toLowerCase();

    // ALWAYS show filename, even if invalid
    setFileName(selectedFile.name);

    // Check invalid formats
    if (!allowedExtensions.includes(ext)) {
      setErrorMessage(
        "⚠ Invalid audio format! Please upload a WAV or MP3"
      );
      setFile(null);

      // Remove preview if exists
      if (audioURL) {
        URL.revokeObjectURL(audioURL);
        setAudioURL("");
      }

      setResult(null);
      return;
    }

    // Valid file
    const newAudioURL = URL.createObjectURL(selectedFile);
    if (audioURL) URL.revokeObjectURL(audioURL);

    setFile(selectedFile);
    setAudioURL(newAudioURL);
    setResult(null);
    setErrorMessage(""); 
  };

  const handleAnalyze = async () => {
  if (!file) {
    setErrorMessage("⚠ Please select a valid audio file!");
    setResult(null);
    return;
  }

  setLoading(true);
  setResult(null);
  setErrorMessage("");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    // 🔥 IMPORTANT FIX
    if (!response.ok) {
      throw new Error("Server response not OK");
    }

    const data = await response.json();

    console.log("✅ Backend Response:", data); // DEBUG

    setLoading(false);

    if (data.error) {
      setErrorMessage("⚠ " + data.error);
    } else {
      setResult(data);
    }

  } catch (err) {
    console.error("❌ Fetch error:", err);
    setLoading(false);
    setErrorMessage("⚠ Server connection error.");
  }
};

  return (
    <div className="App">
      <div className="container">
        <h1>🎧 Sentiment & Emotion Analysis</h1>

        <img
          src={emotionRecognition}
          alt="Emotion Recognition"
          className="main-image"
        />

        <div className="upload-row">
          <label className="custom-file-upload">
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              key={audioURL}
            />
            Choose File
          </label>

          <span className="file-name">{fileName}</span>
        </div>

        {audioURL && (
          <audio key={audioURL} controls className="audio-player">
            <source src={audioURL} />
            Your browser does not support the audio tag.
          </audio>
        )}

        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Audio"}
        </button>

        {errorMessage && (
          <div className="result-box" style={{ background: "#ffeaea" }}>
            <p style={{ color: "#d9534f", margin: 0 }}>{errorMessage}</p>
          </div>
        )}

        {result && (
          <div className="result-box">
            <p>
              🗣 <strong>Transcription:</strong>{" "}
              {result.transcription || "Not available"}
            </p>

            <p>
              💬 <strong>Sentiment:</strong>{" "}
              {result.sentiment ? result.sentiment.toUpperCase() : "N/A"}
            </p>

            <p>
              🎭 <strong>Emotion:</strong>{" "}
              {result.emotion
                ? `${result.emotion.toUpperCase()} ${
                    emotionEmoji[result.emotion.toLowerCase()] || "🙂"
                  }`
                : "N/A"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
