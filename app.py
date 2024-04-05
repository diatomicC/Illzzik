import openai
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import speech_to_text
from tempfile import NamedTemporaryFile

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

import requests

def download_model(url, model_path):
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)

def load_model():
    MODEL_URL = 'https://github.com/diatomicC/Illzzik/blob/main/illzzik_model_trained.hdf5'
    MODEL_PATH = 'illzzik_model_trained.hdf5'
    download_model(MODEL_URL, MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
    
# # Load the model
# def load_model():
#     MODEL_PATH = 'illzzik_model_trained.hdf5'
#     model = tf.keras.models.load_model(MODEL_PATH)
#     return model

model = load_model()


# Sidebar for API Key Input
api_key_input = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

# Button to submit the API key
submit_key = st.sidebar.button("Submit API Key")

# Check if the submit button is pressed
if submit_key:
    if api_key_input:
        # Store the API key in session state
        st.session_state['openai_api_key'] = api_key_input
        st.sidebar.success("API Key set successfully!")
    else:
        st.sidebar.error("Please enter a valid API Key.")

# Usage of the API Key
if 'openai_api_key' in st.session_state and st.session_state['openai_api_key']:
    # Set the API Key for your application
    openai.api_key = st.session_state['openai_api_key']

    # Now you can make OpenAI API calls with the set API key
    # Example: openai.Completion.create(...)

    st.write("API Key is set. You can now interact with OpenAI's services.")
else:
    st.write("Please enter and submit your OpenAI API key to proceed.")



def chat_with_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Correct model name
            messages=[
                {"role": "system", "content": "You are a senior friendly health assistant."},
                {"role": "user", "content": f"Provide a detailed explanation of {text} in a clear and concise manner with markdown structure. Also tell what patient should do right now."}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return str(e)  # For debugging purposes

def calculate_dynamic_height(response_text, base_height=30, characters_per_line=20, line_height=10):
    # Calculate dynamic height based on content length
    num_lines = max(1, len(response_text) / characters_per_line)
    dynamic_height = int(base_height + (num_lines * line_height))
    return dynamic_height

def play_text_to_speech(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    with NamedTemporaryFile(delete=True) as fp:
        tts.save(f'{fp.name}.mp3')
        sound = AudioSegment.from_mp3(f'{fp.name}.mp3')
        play(sound)

# Sidebar menu for navigation
with st.sidebar:
    choose = option_menu("IllZZIK", ["Take picture to check", "Chat with Med assistant!"],
                         icons=['camera', 'chat'],
                         menu_icon="info", default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#FFA500"},
                             "icon": {"color": "#02ab21", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#FF8C00"},
                             "nav-link-selected": {"background-color": "#2f5335"},
                         }
                         )

if choose == "Take picture to check":
        
    # UI
    st.title('IllZZIK: AI Skin diseases analyser')


    warning = f"⚠️ Mind that Test accuracy is 78.06%."

    st.markdown(f"<h3 style='text-align: center; color: black;'>{warning}</h3>", unsafe_allow_html=True)


    # Upload or take a picture
    uploaded_file = st.file_uploader("Upload an image or take a picture below", type=["jpg", "jpeg", "png"])
    captured_image = st.camera_input("Take a picture")

    if uploaded_file is not None or captured_image is not None:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = Image.open(captured_image)

        if image:
            st.image(image, caption='Selected Image', use_column_width=True)
            image = ImageOps.exif_transpose(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img = np.array(image.resize((32, 32))) / 255.0
            img = np.expand_dims(img, axis=0)

            predictions = model.predict(img)
            confidence = 100 * np.max(predictions)
            class_names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
                        'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']
            prediction_message = f"Most likely to be {class_names[np.argmax(predictions)]} with a {confidence:.2f}% confidence."

            # Using HTML to increase font size
            st.markdown(f"<h3 style='text-align: center; color: black;'>{prediction_message}</h3>", unsafe_allow_html=True)


            # Button to get a detailed explanation
            if st.button('See Detail'):
                detail = chat_with_openai(prediction_message)
                st.markdown(detail)
            

    
elif choose == "Chat with Med assistant!":
    st.title('Chat with Our Senior-Friendly Medical Assistant AI')


    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def chat_with_openai(text):
        prompt = f"Assume you are a senior-friendly medical assistant. Provide helpful, easy-to-understand advice. \n\n{text}"
        response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages + [{"role": "user", "content": text}]
        )
        st.session_state.messages.append({"role": "user", "content": text})
        assistant_response = response.choices[0].message["content"]
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Update UI with assistant's response before playing it as speech
        # This is where you display the assistant's response in the UI
        index = len(st.session_state.messages) - 1  # Index of the latest message
        dynamic_height = calculate_dynamic_height(assistant_response)
        st.text_area("Assistant", value=assistant_response, disabled=True, height=dynamic_height, key=f"assistant_response_{index}")
        
        # Now play the assistant's response as speech
        play_text_to_speech(assistant_response)

    state = st.session_state
    if 'text_received' not in state:
        state.text_received = []

    c1, c2 = st.columns(2)
    with c1:
        st.write("Convert speech to text:")
    with c2:
        text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    if text:
        state.text_received.append(text)

    for idx, text in enumerate(state.text_received):
        st.text(text)
        if idx == len(state.text_received) - 1:
            chat_with_openai(text)

    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.text_input("You", value=message["content"], disabled=True, key=f"user_input_{index}")
        else:
            dynamic_height = calculate_dynamic_height(message["content"])
            st.text_area("Assistant", value=message["content"], disabled=True, height=dynamic_height, key=f"assistant_response_{index}")

