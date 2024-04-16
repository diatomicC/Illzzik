# IllZZIK - AI-powered Health Assistant

IllZZIK is a Streamlit application that combines AI-powered technologies to provide two main functionalities: an AI skin disease analyzer and a senior-friendly medical assistant chatbot. This application uses TensorFlow for image analysis and OpenAI's GPT models for text generation.

## Features

1. **AI Skin Disease Analyzer**:
   - Allows users to upload or capture an image of a skin condition.
   - Uses a pre-trained TensorFlow model to predict potential skin diseases.
   - Displays the prediction with confidence levels.

2. **Senior-Friendly Medical Assistant Chatbot**:
   - Users can chat with the assistant using text or voice input.
   - The assistant provides detailed, easy-to-understand medical advice.
   - Uses OpenAI's GPT models to generate responses.

## Installation

To run this application, you need Python installed on your system. Follow these steps:

1. Clone the repository:
   ```bash
   git clone [URL to the repository]
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Setting Up

- Input your OpenAI API key in the sidebar to enable the chatbot functionality.

### AI Skin Disease Analyzer

- Navigate to "Take picture to check" in the sidebar.
- Upload an image or take a picture of the skin condition.
- The system will analyze the image and provide a prediction along with the confidence level.

### Senior-Friendly Medical Assistant

- Navigate to "Chat with Med assistant!" in the sidebar.
- You can interact with the assistant either by typing or by using the speech-to-text feature.
- The assistant responds with detailed advice tailored for senior citizens.

## Note

The AI skin disease analyzer has a test accuracy of 78.06%. It should not be used as a definitive medical diagnosis tool. Always consult with a healthcare professional for medical advice.

## Live Demo

You can access a live demo of the app here: [https://mam7tdt5dmvkk5vcp9vtzg.streamlit.app/](https://mam7tdt5dmvkk5vcp9vtzg.streamlit.app/)
