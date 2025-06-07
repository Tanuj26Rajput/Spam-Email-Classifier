import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model("Spam-Email-Classifier.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict(text):
    seq = tokenizer.texts_to_sequnces([text])
    padded = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
    prediction = model.predict(padded)[0][0]
    return "Spam" if prediction < 0.5 else "Not Spam"

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📩 Spam Email Classifier")
st.markdown("Enter the email content below to check is it's spam or not.")

content = st.text_area("Email Content", placeholder="Type your email content here...")

if st.button("Classify"):
    if content:
        result = predict(content)
        st.success(f"The email is classsified as: {result}")
    else:
        st.error("Please enter the email content to classify.")
