import openai
import streamlit as st
import toml
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import speech_to_text

def calculate_dynamic_height(response_text, base_height=50, characters_per_line=20, line_height=10):
    # Calculate dynamic height for text areas based on content length
    num_lines = max(1, len(response_text) / characters_per_line)
    dynamic_height = int(base_height + (num_lines * line_height))
    return dynamic_height

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

    # Additional content and functionality for AI diagnosis using the picture can be implemented here
    
elif choose == "Chat with Med assistant!":
    st.title('Chat with Our Senior-Friendly Medical Assistant AI')

    # Load OpenAI API key from .streamlit/secrets.toml
    secrets = toml.load(".streamlit/secrets.toml")
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    # Initialize session state variables for chat history and model selection
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def chat_with_openai(text):
        # Construct prompt for a senior-friendly medical assistant
        prompt = f"Assume you are a senior-friendly medical assistant. Provide helpful, easy-to-understand advice. \n\n{text}"
        response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages + [{"role": "user", "content": text}]
        )
        # Append user's message and assistant's response to maintain chat history
        st.session_state.messages.append({"role": "user", "content": text})
        assistant_response = response.choices[0].message["content"]
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Speech to text conversion setup
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

    # Respond to the latest input
    for idx, text in enumerate(state.text_received):
        st.text(text)
        if idx == len(state.text_received) - 1:
            chat_with_openai(text)

    # Display chat history
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.text_input("You", value=message["content"], disabled=True, key=f"user_input_{index}")
        else:
            dynamic_height = calculate_dynamic_height(message["content"])
            st.text_area("Assistant", value=message["content"], disabled=True, height=dynamic_height, key=f"assistant_response_{index}")
