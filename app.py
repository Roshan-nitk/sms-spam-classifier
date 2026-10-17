
import pickle
import streamlit as st
import pandas as pd

# -----------------------------
# Load the trained ML artifacts
# -----------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# -----------------------------
# Configure the page
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# -----------------------------
# Application heading
# -----------------------------
st.title("📩 SMS Spam Detector")

st.write(
    "An ML-powered application that classifies "
    "SMS messages as spam or legitimate."
)

st.info(
    "Enter a message below and click Analyze Message "
    "to see the model's prediction."
)

# -----------------------------
# Example messages
# -----------------------------
st.subheader("Try an example")

example = st.selectbox(
    "Choose a sample message",
    [
        "Choose an example...",
        "Congratulations! You won a free prize. Call now!",
        "Hey, are we meeting at the library at 4 PM?",
        "URGENT! Claim your cash reward now."
    ]
)

# -----------------------------
# SMS input
# -----------------------------
default_message = "" if example == "Choose an example..." else example

message = st.text_area(
    "SMS message",
    value=default_message,
    placeholder="Type or paste your SMS here...",
    height=130
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Message", type="primary", use_container_width=True):

    if not message.strip():
        st.warning("Please enter an SMS message.")

    else:
        # Convert text into dense numerical features
        message_vector = vectorizer.transform([message]).toarray()

        # Predict class and probabilities
        prediction = model.predict(message_vector)[0]
        probabilities = model.predict_proba(message_vector)[0]

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        spam_probability = class_probabilities[1]
        ham_probability = class_probabilities[0]

        # Display prediction result
        st.divider()
        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("🚨 Classification: SPAM")
        else:
            st.success("✅ Classification: HAM (NOT SPAM)")

        # Display probabilities
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam Probability",
                f"{spam_probability:.2%}"
            )

        with col2:
            st.metric(
                "Ham Probability",
                f"{ham_probability:.2%}"
            )

        # Probability chart
        chart_data = pd.DataFrame({
            "Category": ["Spam", "Ham"],
            "Estimated probability": [
                spam_probability,
                ham_probability
            ]
        }).set_index("Category")

        st.subheader("Probability Breakdown")
        st.bar_chart(chart_data)

        st.caption(
            "Probabilities are model estimates, not guarantees. "
            "Do not rely on this tool alone to determine whether "
            "a message is safe."
        )
