import streamlit as st
import ollama

# Définir le titre de l'application
st.title("Emsi Chatbot")

# --- Options de configuration ---
st.sidebar.header("Options de configuration")

# Sélection du modèle
available_models = ["llama3.2", "qwen:0.5b"]  # Liste des modèles disponibles
selected_model = st.sidebar.selectbox("Modèle", available_models, index=0)

# Température
temperature = st.sidebar.slider("Température", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# Top-k
top_k = st.sidebar.slider("Top-k", min_value=1, max_value=100, value=40, step=1)

# Top-p
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

# Nombre maximal de tokens (num_predict)
max_tokens = st.sidebar.slider("Nombre maximal de tokens", min_value=1, max_value=2048, value=512, step=1) # Ajuster la valeur maximale selon le modèle

# --- Fin des options de configuration ---

# Initialiser l'historique des messages si il n'existe pas
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages existants
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accepter l'entrée de l'utilisateur
prompt = st.chat_input("Que voulez-vous me demander ?")
if prompt:
    # Afficher le message de l'utilisateur dans l'interface
    with st.chat_message("user"):
        st.markdown(prompt)
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Obtenir la réponse du modèle
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Appeler le modèle avec les options configurées
        for response in ollama.chat(
            model=selected_model,
            messages=[
                {
                    'role': m['role'],
                    'content': m['content'],
                }
                for m in st.session_state.messages
            ],
            options={
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'num_predict': max_tokens,
            },
            stream=True
        ):
            full_response += response['message']['content']
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Ajouter la réponse du modèle à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})