import openai
import streamlit as st
import toml
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import speech_to_text
from tempfile import NamedTemporaryFile

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
    st.title('Take a Picture for AI Diagnosis')
    picture = st.camera_input("Take a picture")

    # Additional functionality for AI diagnosis using the picture can be implemented here
    
elif choose == "Chat with Med assistant!":
    st.title('Chat with Our Senior-Friendly Medical Assistant AI')

    # Load OpenAI API key from .streamlit/secrets.toml
    secrets = toml.load(".streamlit/secrets.toml")
    openai.api_key = secrets["OPENAI_API_KEY"]

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # def chat_with_openai(text):
    #     prompt = f"Assume you are a senior-friendly medical assistant. Provide helpful, easy-to-understand advice. \n\n{text}"
    #     response = openai.ChatCompletion.create(
    #         model=st.session_state["openai_model"],
    #         messages=st.session_state.messages + [{"role": "user", "content": text}]
    #     )
    #     st.session_state.messages.append({"role": "user", "content": text})
    #     assistant_response = response.choices[0].message["content"]
    #     st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
    #     # Play assistant's response as speech
    #     play_text_to_speech(assistant_response)

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
