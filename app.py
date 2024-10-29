# # Save this code in a file named app.py
# import streamlit as st
# from transformers import pipeline

# # Load the Gemma model for text generation
# model_name = "kingabzpro/Gemma-2-9b-it-chat-doctor"
# chatbot = pipeline("text-generation", model=model_name)

# # Streamlit app layout
# st.title("Health Chatbot")
# st.write("Ask me about your symptoms and I'll suggest possible causes.")

# # User input
# user_input = st.text_input("Enter your symptoms:")

# if user_input:
#     response = chatbot(
#         user_input,
#         max_length=150,  # Adjust max length as needed
#         num_return_sequences=1,
#         truncation=True,
#         pad_token_id=50256
#     )
#     st.write(f"Bot: {response[0]['generated_text']}")





import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model function
@st.cache_resource
def load_model():
    model_name = "kingabzpro/Gemma-2-9b-it-chat-doctor"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the model
chatbot = load_model()

# Streamlit app title
st.title("Health Chatbot")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.text_area("Chat History:", chat, height=100, disabled=True)

# User input for the chatbot
user_input = st.text_input("Ask your health-related question:")

if st.button("Get Response"):
    if user_input:
        # Add user question to chat history
        st.session_state.chat_history.append(f"You: {user_input}")
        
        # Generate response from the chatbot
        response = chatbot(user_input, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        # Add bot response to chat history
        st.session_state.chat_history.append(f"Bot: {response}")
        
        # Display the bot response
        st.text_area("Chatbot Response:", response, height=200, disabled=True)
    else:
        st.warning("Please enter a question.")

# Optional: Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    st.success("Chat history cleared!")
