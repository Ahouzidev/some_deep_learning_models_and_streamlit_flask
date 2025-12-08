"""
API Flask - Conversion de votre Streamlit Gemini Chatbot
À exécuter dans PyCharm en parallèle de votre projet Flutter
"""
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis Flutter
load_dotenv()
# Initialize Gemini Client (même logique que votre Streamlit)
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini API Key configurée avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de Gemini: {e}")
        client = None
else:
    client = None
    print("⚠️  GEMINI_API_KEY non trouvée dans les variables d'environnement")
    print("💡 Solution: Créez un fichier .env avec GEMINI_API_KEY=votre_cle")

# ==========================================
# 2. STOCKAGE DES SESSIONS (équivalent st.session_state)
# ==========================================
# Dictionnaire pour stocker les messages de chaque utilisateur
user_conversations = {}


# ==========================================
# 3. ROUTES API
# ==========================================

@app.route('/')
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        'message': '🤖 Gemini Chatbot API ',
        'version': '1.0',
        'status': 'online' if client else 'api_key_missing',
        'model': 'gemini-2.5-flash',
        'endpoints': {
            'POST /api/chat': 'Envoyer un message au chatbot',
            'POST /api/clear': 'Effacer l\'historique d\'un utilisateur',
            'GET /api/history/<user_id>': 'Récupérer l\'historique d\'un utilisateur',
            'GET /health': 'Vérifier l\'état du serveur'
        }
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint principal pour le chatbot
    Reproduction exacte de votre logique Streamlit
    """

    # Vérifier que Gemini est configuré
    if not client:
        return jsonify({
            'error': 'Gemini API Key not found. Cannot send messages.',
            'response': 'Erreur de configuration du serveur. Contactez l\'administrateur.',
            'status': 'error'
        }), 500

    try:
        # Récupérer les données envoyées par Flutter
        data = request.json
        prompt = data.get('message', '').strip()
        user_id = data.get('user_id', 'anonymous')
        user_name = data.get('user_name', 'Utilisateur')

        # Validation
        if not prompt:
            return jsonify({
                'error': 'Message vide',
                'response': 'Veuillez entrer un message.',
                'status': 'error'
            }), 400

        # Initialiser les messages de l'utilisateur (équivalent st.session_state)
        if user_id not in user_conversations:
            user_conversations[user_id] = []

        # Sauvegarder le message de l'utilisateur
        user_conversations[user_id].append({
            "role": "user",
            "text": prompt
        })

        # Préparer le prompt complet (vous pouvez ajouter des instructions système ici)
        full_prompt = prompt

        # Optionnel: Ajouter un contexte système pour EMSI
        system_instruction = f"""Tu es un assistant virtuel polyvalent.
        Tu aides {user_name} avec ses questions académiques, professionnelles ou personnelles, ainsi que toute information utile dans différents domaines.
        Réponds de manière claire, professionnelle et amicale, en t'adaptant à la langue et au contexte de l'utilisateur."""

        # Construire le contexte avec l'historique (optionnel)
        context = system_instruction + "\n\n"
        for msg in user_conversations[user_id][-10:]:  # Garder les 10 derniers messages
            context += f"{msg['role']}: {msg['text']}\n"

        # ⭐ APPEL À GEMINI (exactement comme dans votre Streamlit)
        print(f"📤 Envoi à Gemini pour {user_name}: {prompt[:50]}...")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=context  # ou full_prompt si vous ne voulez pas d'historique
        )

        bot_reply = response.text

        print(f"📥 Réponse reçue: {bot_reply[:50]}...")

        # Sauvegarder la réponse de l'assistant
        user_conversations[user_id].append({
            "role": "assistant",
            "text": bot_reply
        })

        # Limiter l'historique à 20 messages (10 échanges)
        if len(user_conversations[user_id]) > 20:
            user_conversations[user_id] = user_conversations[user_id][-20:]

        # Retourner la réponse à Flutter
        return jsonify({
            'response': bot_reply,
            'status': 'success',
            'message_count': len(user_conversations[user_id])
        })

    except Exception as e:
        error_message = str(e)
        print(f"❌ Erreur: {error_message}")

        return jsonify({
            'error': error_message,
            'response': f'Désolé, une erreur est survenue: {error_message}',
            'status': 'error'
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Effacer l'historique d'un utilisateur"""
    try:
        data = request.json
        user_id = data.get('user_id', 'anonymous')

        if user_id in user_conversations:
            user_conversations[user_id] = []
            print(f"🗑️  Historique effacé pour l'utilisateur: {user_id}")

        return jsonify({
            'status': 'success',
            'message': 'Conversation effacée avec succès'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Récupérer l'historique complet d'un utilisateur"""
    if user_id in user_conversations:
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'messages': user_conversations[user_id],
            'count': len(user_conversations[user_id])
        })
    else:
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'messages': [],
            'count': 0
        })


@app.route('/health')
def health():
    """Vérifier l'état du serveur"""
    total_messages = sum(len(messages) for messages in user_conversations.values())

    return jsonify({
        'status': 'online',
        'gemini_configured': client is not None,
        'active_users': len(user_conversations),
        'total_messages': total_messages,
        'model': 'gemini-2.5-flash'
    })


# ==========================================
# 4. LANCEMENT DU SERVEUR
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 SERVEUR API GEMINI CHATBOT - EMSI")
    print("=" * 70)
    print(f"📍 URL du serveur: http://localhost:8000")
    print(f"✅ Gemini API: {'Configurée ✓' if client else '❌ Non configurée'}")
    print(f"📝 Modèle: gemini-2.5-flash")
    print(f"⏹️  Appuyez sur Ctrl+C pour arrêter le serveur")
    print("=" * 70 + "\n")

    # Démarrer le serveur Flask
    app.run(
        host='0.0.0.0',  # Accessible depuis l'extérieur
        port=8000,  # Port 8000 (différent de Streamlit qui est sur 8501)
        debug=True  # Mode debug pour voir les erreurs
    )