# STREET Whisper Transcription App

## Structure
```backend```: A folder that contains all files related to the backend of the application such as scripts that does translation, transcription, etc. 
```streetwhisperapp.py```: Script for the command line interface of the application.
```requirements.txt```: All the dependencies that needs to be installed to run the app.

## How To Run App
This app currently runs on ~Python 3.9. Please be sure to have Python 3.9 installed on your device. You can install Python 3.9 from here: https://www.python.org/downloads/. 

### For MacOS 
1. Accept the conditions for ```speaker-diarization-3.1``` on Hugging Face, which can be found here: https://huggingface.co/pyannote/speaker-diarization-3.1 and then head to your Hugging Face account's settings, then to the Access Tokens section, and then create a New Token (the type can be either Read or Write but we recommend you setting the type to Read). You will need to enter this token the first time you run the app. 
2. Open your terminal. This can be found by going to launchpad and searching for "terminal".
3. To run the app locally:
    1. ```cd``` is a command used that will allow you to move between directories. We will use the command ```cd``` to eventually end up in the street-whisper-app folder. To do this locate where the street-whisper-app folder is on your computer. 
        - Example: Say the ```street-whisper-app``` folder is located in your GitHub folder, which is in your documents folder on your computer, this would be the order of commands you will enter (Note: you enter after writing every command): ```cd documents -> cd GitHub -> cd street-whisper-app```.
    2. In the ```street-whisper-app``` folder, if this is your first time running the app we want to create a virtual environment, which is an isolated place on your computer with all the dependencies we need to run the app. To create the virtual environment (assuming Python 3.9 has been installed), run the following command with <virtual-environment-name> replaced with the name of your choosing (ex: ```python3.9 -m venv myvenv```): ```python3.9 -m venv <virtual-environment-name>```. To activate the virtual environment, use the following command: ```source <virtual-environment-name>/bin/activate```. In other times when you run the app, you do not need to create a new virtual environment again and you can use ```source <virtual-environment-name>/bin/activate``` to activate the virtual environment. 
    5. Now we will need to install the necessary dependencies if you have just created a new virtual environment (NOTE: you only need to install dependencies once if you do not create a new virtual environment and use the same one). To do this, run the following command: ```pip install -r requirements.txt``` and then to install Whisper in the virtual environment, run the following: ```pip install -q git+https://github.com/openai/whisper.git > /dev/null```
    6. From there enter the following command: ```python streetwhisperapp.py start``` to start the app.
    7. If this is your first time running the app on your device, enter the access token from Hugging Face when prompted by the app. If you have entered a valid token in the past, you can just click enter when the app ask for an access token.

Note: The app can run offline after a valid access token has been entered. 

Link to the original repository containing the original scripts for the models: https://github.com/carmen-chau/StreetWhisperCode
