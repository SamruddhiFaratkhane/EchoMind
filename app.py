import streamlit as st
import spacy
from models.sentiment_model import load_sentiment_model, get_sentiment
from utils.text_preprocessing import clean_text
from speech.transcriber import transcribe_audio
import pandas as pd
from datetime import datetime
import random
import altair as alt

# --- Page config ---
st.set_page_config(page_title="EchoMind – Your AI-Powered Emotional Mirror", layout="centered")

st.markdown(
    """
    <style>
    /* Page background and main container */
    .main {
        background-color: #f0f4f8;
        padding: 30px 40px;
        border-radius: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #263238;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #d0e2ff;
        padding: 20px 15px;
        border-radius: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stSidebar"] h2 {
        color: #1e3a8a;
        font-weight: 700;
    }
    [data-testid="stSidebar"] p {
        font-size: 14px;
        color: #374151;
        line-height: 1.5;
    }

    /* TextArea styling */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 12px;
        border: 2px solid #cfd8dc;
        padding: 15px;
        resize: vertical;
        min-height: 180px;
        box-shadow: 0 2px 4px rgb(0 0 0 / 0.1);
        color: #263238;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #4a90e2;
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 30px;
        border: none;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(74, 144, 226, 0.4);
        cursor: pointer;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #357ABD;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        color: #263238;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load models once ---
sentiment_model = load_sentiment_model()
nlp = spacy.load("en_core_web_sm")

# --- Helper function for NER ---
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# --- Logging function ---
def log_assessment(text, sentiment_label, sentiment_score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "timestamp": timestamp,
        "text": text,
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score * 100, 2)
    }
    try:
        df = pd.read_csv("assessment_logs.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["timestamp", "text", "sentiment_label", "sentiment_score"])

    new_row = pd.DataFrame([log_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("assessment_logs.csv", index=False)

# --- Motivational quotes ---
def get_quote(sentiment_label):
    quotes = {
        "POSITIVE": [
            "Keep shining, you're doing great!",
            "Every day is a fresh start. Keep going!",
            "Happiness looks good on you!"
        ],
        "NEGATIVE": [
            "You seem down. Try talking to someone, or take a break 💛",
            "This too shall pass. Stay strong!",
            "Remember, self-care is not selfish."
        ],
        "NEUTRAL": [
            "You got this, dear 🌸",
            "Don't worry, eventually everything will make sense ✨"
        ]
    }
    return random.choice(quotes.get(sentiment_label, ["Keep going, don't stop!"]))

# --- Load existing logs for mood chart ---
try:
    df_logs = pd.read_csv("assessment_logs.csv")
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
except FileNotFoundError:
    df_logs = pd.DataFrame(columns=["timestamp", "text", "sentiment_label", "sentiment_score"])

# --- Sidebar with info and tips ---
with st.sidebar:
    st.header("🛠️ Settings & Tips")
    show_advanced = st.checkbox("Show advanced options")

    st.markdown(
        """
        **How to get the best results:**

        - Be honest with your inputs ✍️  
        - Take a moment to reflect 🧘‍♀️  
        - Use short sentences for clarity 🗣️  
        
        ---
        **Mood tips:**
        - 😊 Smile more, it boosts mood!  
        - 🌿 Try a short walk in nature.  
        - 🎶 Listen to uplifting music.  
        """
    )

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🎙️ Audio Upload", "📈 Mood Chart"])

with tab1:
    st.header("✍️ Express Yourself")
    user_input = st.text_area("Enter your thoughts or feelings:")

    if st.button("Analyze Text"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            clean = clean_text(user_input)

        import nltk
        nltk.download('punkt')

        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(user_input)
        results = []

        for sentence in sentences:
            clean_sent = clean_text(sentence)
            sent_label, sent_score = get_sentiment(clean_sent, sentiment_model)
            results.append((sentence, sent_label, round(sent_score * 100, 2)))

        # Display individual sentence sentiment
        st.subheader("🧾 Sentence-wise Sentiment Breakdown")
        for i, (sentence, label, score) in enumerate(results, 1):
            st.write(f"**{i}.** {sentence}")
            st.write(f"→ Sentiment: `{label}` ({score}%)")
            st.markdown("---")

        # Derive overall sentiment based on dominant label
        label_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for _, label, _ in results:
            label_counts[label] += 1

        overall_label = max(label_counts, key=label_counts.get)
        avg_score = round(sum(score for _, _, score in results) / len(results), 2)

        #st.success(f"**Overall Sentiment (Sentence-wise):** {overall_label} ({avg_score}%)")

        # ----- FULL TEXT sentiment analysis -----
        full_clean = clean_text(user_input)
        full_label, full_score = get_sentiment(full_clean, sentiment_model)
        st.info(f"**Full Text Sentiment:** {full_label} ({full_score * 100:.2f}%)")

        # Show motivational quote based on sentence-wise overall
        quote = get_quote(full_label)
        st.info(quote)

        # Show balloons only if Full Text Sentiment is POSITIVE
        if full_label == "POSITIVE":
            st.balloons()

        # Log the full input (not sentence-wise) for history
        log_assessment(user_input, overall_label, avg_score)

        # Named Entities
        st.subheader("🔍 Named Entities (from your text)")
        entities = extract_entities(user_input)
        if entities:
            for ent_text, ent_label in entities:
                st.write(f"- {ent_text} ({ent_label})")
        else:
            st.write("No named entities found.")


with tab2:
    st.header("🎙️ Speak Your Mind")
    audio_file = st.file_uploader("Upload your voice recording (.mp3 or .wav):", type=["mp3", "wav"])

    if audio_file:
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.read())

        st.info("Transcribing audio... please wait.")
        transcribed_text = transcribe_audio("temp_audio.mp3")
        st.success("Transcription complete!")

        st.subheader("📝 Transcribed Text")
        st.write(transcribed_text)

        clean = clean_text(transcribed_text)
        label, score = get_sentiment(clean, sentiment_model)
        st.subheader("💬 Sentiment from Audio")
        st.write(f"🧠 Sentiment: **{label}** ({score * 100:.2f}%)")

        quote = get_quote(label)
        st.info(quote)

        st.subheader("🔍 Named Entities (from audio)")
        entities = extract_entities(transcribed_text)
        if entities:
            for ent_text, ent_label in entities:
                st.write(f"- {ent_text} ({ent_label})")
        else:
            st.write("No named entities found.")

        # Log the result
        log_assessment(transcribed_text, label, score)

with tab3:
    st.header("Mood Trend Over Time")
    if not df_logs.empty:
        chart = alt.Chart(df_logs).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title='Date'),
            y=alt.Y('sentiment_score:Q', title='Sentiment Score (%)'),
            color=alt.Color('sentiment_label:N', title='Sentiment'),
            tooltip=['timestamp:T', 'sentiment_score:Q', 'sentiment_label:N']
        ).properties(
            width=700,
            height=400,
            title="📈 Mood Trend Over Time"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to show mood trends yet.")

# --- Advanced options placeholder ---
if show_advanced:
    st.info("Advanced options will be added soon! Stay tuned 😊")
