import streamlit as st
from google import genai # make sure python-genai is installed
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")  # or set in Streamlit secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None
    st.error("Gemini API Key is missing. Set it in environment variables or Streamlit secrets.")

# ==========================================
# 2. SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.title("🤖 Gemini Chatbot")
st.markdown("Start chatting with your AI assistant!")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# User input
if prompt := st.chat_input("Type your message..."):
    if not client:
        st.error("Gemini API Key not found. Cannot send messages.")
    else:
        # Save user message
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare full prompt (could add system instructions here)
        full_prompt = prompt

        # Send to Gemini
        try:
            with st.spinner("Gemini is thinking..."):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                bot_reply = response.text

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(bot_reply)

            # Save assistant message
            st.session_state.messages.append({"role": "assistant", "text": bot_reply})

        except Exception as e:
            st.error(f"An error occurred: {e}")
