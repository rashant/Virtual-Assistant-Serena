import speech_recognition as sr
import pyttsx3
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
engine.setProperty('rate', 195)


def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    try:
        with sr.Microphone() as source:
            listener.dynamic_energy_threshold = False
            listener.energy_threshold = 2300
            print('listening...')
            voice = listener.listen(source, timeout=None, phrase_time_limit=None)
            command = listener.recognize_google(voice)
            command = command.lower()
            print('recognizing...')
            if 'erica' in command or 'erika' in command:
                command = command.replace('erika', '')
                command = command.replace('erica', '')
            return command
    except:
        return 'None'
