import gradio as gr
import time
import re 
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import json
import requests
import os

# --- Gemini API Configuration ---
# API Key is left blank. The system will inject it at runtime in the fetch call.
GEMINI_API_KEY = "bjWwMUB979cCwLHEwFD9dlaG8"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- 1. Training Data (Defining Intents) ---

# Added a new intent 'general_health_llm' to route non-symptom-reporting queries directly to the LLM.
# I have added more phrases to 'general_health_llm' to strengthen its classification against 'symptom_inquiry'.
data = {
    'intent': [
        'greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
        'schedule_appointment', 'schedule_appointment', 'schedule_appointment', 'schedule_appointment',
        'visiting_hours', 'visiting_hours', 'visiting_hours', 'visiting_hours',
        'symptom_inquiry', 'symptom_inquiry', 'symptom_inquiry', 'symptom_inquiry',
        'general_health_llm', 'general_health_llm', 'general_health_llm', 'general_health_llm', 
        'general_health_llm', 'general_health_llm', 'general_health_llm', 'general_health_llm', 
        'general_health_llm', 'general_health_llm', 'general_health_llm', 'general_health_llm',
        'general_health_llm', 'general_health_llm'
    ],
    'text': [
        'hello', 'hi there', 'good morning', 'hey', 'can you help me', 'I have a question', 
        'what is up', 'just saying hi', 'hey bro',
        'I need to book an appointment', 'Can I see a doctor tomorrow', 'schedule a checkup', 'book me in on Tuesday',
        'What are the visiting hours', 'When can I visit a patient', 'visitor hours please', 'what time do you open',
        'I have a cough', 'My head hurts really bad', 'I feel sick', 'what to do for a fever',
        'What are the symptoms of a common cold?', 'how does a vaccine work', 'tell me about the flu shot', 'what is the treatment for a broken bone', 
        'What is COVID-19', 'Define hypertension', 'How long does the flu last', 'Is coffee good for me', # Added more phrases
        'What is the function of the heart', 'Tell me about viruses', 'When was aspirin invented', 'explain medical imaging', # Added more phrases
        'How often should I exercise?', 'what is cholesterol' # Added final phrases
    ]
}

# Convert the dictionary into a Pandas DataFrame.
df = pd.DataFrame(data)

# --- 2. Model Training (A simple Text Classifier) ---
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(df['text'])
y_train = df['intent']

classifier = LinearSVC()
classifier.fit(X_train, y_train)


# --- LLM Integration Function ---

def generate_llm_response(prompt):
    """Calls the Gemini API to get a natural language response."""
    
    # System instruction to guide the LLM's persona and safety
    system_prompt = (
        "You are a friendly, non-diagnostic, informational virtual health assistant. "
        "Your primary goal is to answer general, common knowledge health or organizational questions. "
        "ALWAYS include a strong, clear disclaimer that you are not a doctor and cannot provide "
        "diagnosis or treatment advice, and suggest the user see a qualified medical professional. "
        "Keep your answers concise and helpful."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}] # Enable Google Search Grounding for current info
    }

    try:
        # Use a retry mechanism with exponential backoff for robustness
        MAX_RETRIES = 3
        for i in range(MAX_RETRIES):
            response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text
            
            # If rate limited or transient error, wait and retry
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** i) # Exponential backoff
                continue
            
            # Handle non-retryable errors
            return f"Error connecting to LLM: {response.status_code} - {response.text}"

        return "LLM service unavailable after multiple retries."

    except Exception as e:
        return f"An exception occurred during API call: {e}"


# --- 3. Chatbot Logic Function ---

def get_intent(text):
    """Predicts the intent of the user's message using the trained classifier."""
    
    # 1. Check for intentionally empty/very short input first (robustness)
    if not text or len(text.strip()) < 2:
        return 'fallback' 
        
    # 2. Predict the intent
    X_test = vectorizer.transform([text.lower()])
    prediction = classifier.predict(X_test)[0]
    
    return prediction

def respond(message, chat_history):
    """The main function that generates the bot's response based on intent."""
    
    # Add user message to history immediately
    chat_history.append((message, None))

    # Get predicted intent
    intent = get_intent(message)
    bot_response = ""

    # --- Conversation Flow Logic ---
    if intent == 'greeting':
        responses = [
            "Hello! I'm your virtual patient support assistant. How can I help you today?",
            "Hi there! Welcome. Are you looking to schedule an appointment or ask a general question?",
            "Good to hear from you! What brings you here? (e.g., 'Book an appointment' or 'Visiting hours')"
        ]
        bot_response = random.choice(responses)

    elif intent == 'schedule_appointment':
        # Hard-coded, reliable action response
        date_match = re.search(r'(today|tomorrow|monday|tuesday|wednesday|thursday|friday|weekend|\d{1,2}/\d{1,2})', message.lower())
        
        if date_match:
            date_info = date_match.group(0).capitalize()
            bot_response = f"I see you want to book for **{date_info}**. That's great! To confirm, what time works best for you? (This part would normally connect to a scheduling API.)"
        else:
            bot_response = "I can help you book an appointment. What day and time would you prefer to come in?"

    elif intent == 'visiting_hours':
        # Hard-coded, reliable factual response
        bot_response = "The general visiting hours are Monday to Friday from **9:00 AM to 5:00 PM**. For ICU or specialized unit hours, please check our main website. Can I help you with anything else?"

    elif intent == 'symptom_inquiry':
        # Hard-coded, critical safety response for users reporting current symptoms (e.g., "I have a cough")
        bot_response = (
            "⚠️ **Disclaimer:** I am a prototype and cannot offer medical advice or diagnosis. "
            "**If this is an emergency, please call 911 or your local emergency number.** "
            "For non-emergency concerns, I strongly recommend you **schedule an appointment** with a physician. Shall I help you book one?"
        )
    
    elif intent == 'general_health_llm': # NEW: Route to LLM for non-critical knowledge
        # --- LLM Escalation for General/Uncertain Queries ---
        time.sleep(0.5)
        llm_response = generate_llm_response(message)
        bot_response = llm_response

    else: # Fallback - only happens if LLM is also not confident or intent is completely unknown
        # --- LLM Escalation for True Fallback ---
        
        # Simulate thinking delay before API call
        time.sleep(0.5)
        
        # We'll use the LLM to generate a helpful response, even for fallback
        llm_response = generate_llm_response(message)
        bot_response = llm_response

    # Simulate typing delay for a smoother user experience
    time.sleep(1.0)
    
    # Append bot's response to history
    chat_history[-1] = (message, bot_response)
    
    return chat_history

# --- 4. Gradio Interface Definition ---

with gr.Blocks(title="Patient Support Chatbot Prototype") as demo:
    gr.Markdown(
        """
        # 🤖 Hybrid Patient Support Chatbot Prototype (ML + LLM)
        This prototype uses a **Machine Learning Classifier** for reliable actions (scheduling) 
        and an **LLM (Gemini)** for general knowledge and conversational questions.
        
        **Try asking:** 1. **ML Action:** `I need to book an appointment for tomorrow`
        2. **General Health (LLM):** `What are the symptoms of a common cold?`
        3. **Symptom Inquiry (Safety):** `I feel sick` (This should still trigger the warning)
        """
    )

    # Chatbot component stores the history
    chatbot = gr.Chatbot(
        label="Virtual Health Assistant",
        height=400,
        avatar_images=(None, "https://placehold.co/60x60/34D399/white?text=AI") # Custom avatar for flair
    )

    # Chat input component
    msg = gr.Textbox(
        label="Your Message", 
        placeholder="Type your question here...",
        scale=4
    )
    
    # Clear button
    clear = gr.ClearButton([msg, chatbot], value="Clear Chat")

    # The action that happens when the user hits enter or clicks submit
    msg.submit(respond, [msg, chatbot], chatbot)

    # Clear the input box after submission
    msg.submit(lambda: "", None, msg) 


# Set launch options (optional, remove the next line if you are not running in an environment that supports Gradio launch)
demo.launch()
