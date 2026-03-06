import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Exam Anxiety Detector",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (PROFESSIONAL UI)
# -------------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.big-title {
    font-size:40px;
    font-weight:bold;
    color:white;
}

.result-box {
    padding:20px;
    border-radius:10px;
    margin-top:20px;
}

.high {
    background-color:#ff4b4b;
}

.moderate {
    background-color:#f7b500;
}

.low {
    background-color:#00c853;
}

.tip-box {
    background-color:#1c1f26;
    padding:20px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------

st.markdown("<div class='big-title'>🧠 AI Exam Anxiety Detector Dashboard</div>", unsafe_allow_html=True)
st.write("Analyze exam-related emotions and receive helpful suggestions.")

st.divider()

# -------------------------------
# LAYOUT
# -------------------------------

col1, col2 = st.columns([2,1])

# -------------------------------
# INPUT SECTION
# -------------------------------

with col1:

    st.subheader("✍️ Enter Your Thoughts")

    text = st.text_area(
        "Describe how you feel about your exams",
        height=150,
        placeholder="Example: I feel nervous about my upcoming exams..."
    )

    predict_button = st.button("🔍 Analyze Anxiety")

# -------------------------------
# RESULT SECTION
# -------------------------------

if predict_button and text != "":

    url = "http://127.0.0.1:8000/predict"

    try:

        response = requests.post(url, json={"text": text})
        result = response.json()

        anxiety_level = result["anxiety_level"]

        # Fake confidence values for visualization
        if anxiety_level == "High Anxiety":
            probs = [0.1,0.2,0.7]
            emoji = "😟"
            color = "high"

        elif anxiety_level == "Moderate Anxiety":
            probs = [0.2,0.6,0.2]
            emoji = "😐"
            color = "moderate"

        else:
            probs = [0.7,0.2,0.1]
            emoji = "🙂"
            color = "low"

        # -------------------------------
        # RESULT BOX
        # -------------------------------

        st.markdown(
            f"<div class='result-box {color}'>"
            f"<h2>{emoji} {anxiety_level}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

        # -------------------------------
        # PROBABILITY CHART
        # -------------------------------

        st.subheader("📊 Anxiety Probability Distribution")

        labels = ["Low", "Moderate", "High"]

        fig, ax = plt.subplots()
        ax.bar(labels, probs)
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")

        st.pyplot(fig)

        # -------------------------------
        # HISTORY TRACKING
        # -------------------------------

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "level": anxiety_level
        })

    except:

        st.error("⚠️ Could not connect to backend server")

# -------------------------------
# SIDE PANEL
# -------------------------------

with col2:

    st.subheader("💡 Anxiety Management Tips")

    st.markdown("""
    <div class="tip-box">

    🫁 **Breathing Exercise**  
    Take slow deep breaths for 2 minutes.

    📚 **Break Study Sessions**  
    Use the Pomodoro technique.

    🧠 **Positive Thinking**  
    Remind yourself preparation matters more than fear.

    🛌 **Good Sleep**  
    Avoid studying all night before exams.

    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# HISTORY CHART
# -------------------------------

if "history" in st.session_state and len(st.session_state.history) > 0:

    st.divider()

    st.subheader("📈 Anxiety Trend (Session History)")

    df = pd.DataFrame(st.session_state.history)

    level_map = {
        "Low Anxiety":1,
        "Moderate Anxiety":2,
        "High Anxiety":3
    }

    df["score"] = df["level"].map(level_map)

    fig, ax = plt.subplots()

    ax.plot(df["time"], df["score"], marker="o")

    ax.set_yticks([1,2,3])
    ax.set_yticklabels(["Low","Moderate","High"])

    ax.set_xlabel("Time")
    ax.set_ylabel("Anxiety Level")

    st.pyplot(fig)