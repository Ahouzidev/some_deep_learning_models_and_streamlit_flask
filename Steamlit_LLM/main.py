import streamlit as st
from openai import OpenAI

# -----------------------------
# Configure OpenAI API client
# -----------------------------


MODEL_NAME = "gpt-4o-mini"  # fast + cheap + very good

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GPT Chatbot", page_icon="🤖")
st.title("🤖 GPT Chatbot (Streamlit)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate GPT response
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *st.session_state.messages
                ]
            )
            reply = response.choices[0].message.content

        except Exception as e:
            reply = f"Error: {e}"

        st.write(reply)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
