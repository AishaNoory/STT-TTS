Lightweight Voice Assistant
Overview
This project is a lightweight voice assistant application that:

Recognizes Speech (STT): Converts spoken audio input into text using the facebook/wav2vec2-base-960h model.

Processes the Text: Generates a response using the distilgpt2 language model.

Converts Text to Speech (TTS): Converts the generated text response back into speech using pyttsx3.



Features
Speech-to-Text (STT): Converts audio input into text.

Text Processing: Generates responses to the input text.

Text-to-Speech (TTS): Converts the generated text back into speech.



Installation
To run this project, you'll need to install the required dependencies. You can do this using pip:

sh
pip install torch transformers gradio pyttsx3 numpy scipy soundfile



Usage
Clone the repository:
git clone <repository_url>
cd <repository_directory>


Run the application:
python <filename>.py


Interact with the voice assistant:

Record your voice using the provided Gradio interface.

The application will transcribe your speech, generate a response, and convert it back to speech.

The transcribed text, generated response, and TTS output will be displayed in the interface.




Code Explanation
Initialization
Initialize the necessary devices and models for STT, text processing, and TTS.

STT Function (transcribe)
This function takes audio input, processes it, and converts it to text using the facebook/wav2vec2-base-960h model.

Response Generation Function (generate_response)
This function generates a response to the transcribed text using the distilgpt2 model. It includes predefined responses for common phrases and a controlled prompt for other inputs.

TTS Function (synthesize_speech)
This function converts the generated text response to speech using pyttsx3.

Gradio Interface
A simple Gradio interface allows users to interact with the voice assistant by recording their voice and receiving audio responses.

Example Interaction
User: "Hello"

Assistant: "Hello! How can I help you today?"

Storage
Save the code file: Save the provided code as voice_assistant.py or any other filename of your choice.

Create a repository: Store the code in a version control system like Git. You can create a repository on platforms like GitHub, GitLab, or Bitbucket.

Include the README: Make sure to include this README file in the root directory of your repository.

Contributing
Feel free to contribute to this project by submitting issues or pull requests. Your contributions are always welcome!

License
This project is licensed under the MIT License. See the LICENSE file for more details.