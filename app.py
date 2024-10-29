# Save this code in a file named app.py
import streamlit as st
from transformers import pipeline

# Load the Gemma model for text generation
model_name = "kingabzpro/Gemma-2-9b-it-chat-doctor"
chatbot = pipeline("text-generation", model=model_name)

# Streamlit app layout
st.title("Health Chatbot")
st.write("Ask me about your symptoms and I'll suggest possible causes.")

# User input
user_input = st.text_input("Enter your symptoms:")

if user_input:
    response = chatbot(
        user_input,
        max_length=150,  # Adjust max length as needed
        num_return_sequences=1,
        truncation=True,
        pad_token_id=50256
    )
    st.write(f"Bot: {response[0]['generated_text']}")
