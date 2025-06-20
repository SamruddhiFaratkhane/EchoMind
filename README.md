# 🎧 EchoMind – Your AI-Powered Emotional Mirror

**EchoMind** is a Streamlit-based self-assessment tool that helps users reflect on their thoughts and emotions through natural language processing and voice transcription. Whether you're journaling or speaking your feelings, EchoMind analyzes your mood, extracts key emotional cues, and supports your mental well-being.

---

## ✨ Features

- 📝 **Text-Based Journaling** – Type your thoughts and get real-time sentiment feedback
- 🎙️ **Voice Analysis** – Upload MP3/WAV voice recordings for transcription + sentiment detection
- 💬 **Named Entity Recognition** – Understand people, places, and dates in your thoughts
- 📈 **Mood Tracker** – See how your emotional state changes over time
- 💡 **Motivational Quotes** – Personalized based on your mood
- 🧠 **Clean UI** – Built with Streamlit and styled with custom CSS

---

## 📦 How to Run

1. Clone the repo:
   git clone https://github.com/YOUR_USERNAME/echomind.git
   cd echomind

2. Install dependencies:
   pip install -r requirements.txt

3. Download spaCy English model:
   python -m spacy download en_core_web_sm

4. Run the app:
   streamlit run app.py

---

## 📁 Project Structure

echomind/
├── app.py
├── models/
│   └── sentiment_model.py
├── speech/
│   └── transcriber.py
├── utils/
│   └── text_preprocessing.py
├── assessment_logs.csv
├── requirements.txt
├── README.md
└── .gitignore

---

## 💡 Developed By
Samruddhi Faratkhane
| AI & DS, AISSMS IOIT 
| Pune, India
