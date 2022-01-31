import datetime
import math
import os
import random
import sys
from mutagen.mp3 import MP3
import vlc
import threading
import webbrowser
from cmath import sqrt
from time import sleep
import cv2
import face_recognition
import mysql.connector
import numpy as np
import pandas as pd
import pyautogui
import pyjokes
import pyttsx3
import pywhatkit
import requests
import requests_html
import speech_recognition as sr
import wikipedia
from GoogleNews import GoogleNews
from PIL import ImageGrab
from PyDictionary import PyDictionary
from bs4 import BeautifulSoup
from googletrans import Translator
from gtts import gTTS
from win32api import GetSystemMetrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from pygame import mixer

mixer.init()
music_dir = 'Playlist'
song = os.listdir(music_dir)
playlist=[]
for i in song:
    playlist.append('Playlist'+'/'+i)

dictionary = PyDictionary()
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
engine.setProperty('rate', 195)

asking_list = ['anything more', 'will there be anything else', 'still how can i help you', 'still anything to do',
               'any other task', 'what next', 'something else', 'would you like to ask anything else', 'another one',
               'is there anything else', 'more than that', 'something more', 'any other request', 'other stuff']


def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            listener.dynamic_energy_threshold = False
            listener.energy_threshold = 2600
            print('listening...')
            voice = listener.listen(source, timeout=None, phrase_time_limit=None)
            command = listener.recognize_google(voice)
            command = command.lower()
            print('recognizing...')
            if 'serena' in command:
                command = command.replace('serena', '')
            return command
    except:
        pass


def run_serena():
    command = take_command()
    print(command)

    # quit
    if 'see you' in command or 'talk to you later' in command or 'go' in command or ' bye' in command or 'nothing' in command or 'catch you later' in command or 'meet later' in command or 'i am out' in command:
        talk('bye , i will be waiting to do any help. Have a nice time , take care')
        sys.exit()

    # play songs from youtube
    elif 'youtube' in command:
        song = command.replace('play', '')
        print(song)
        pywhatkit.playonyt(song)
        anything()

    # TELL THE CURRENT TIME
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M:%S %p')
        print('time= ' + time)
        talk('The current time is ' + time)
        anything()

    # english dictionary
    elif 'meaning' in command:
        command = command.split(' ')[-1]
        meaning = dictionary.meaning(f"{command}".capitalize())
        k = 1
        for i in meaning.values():
            for j in i:
                print(j)
                talk(j)
                if k == 5:
                    break
                else:
                    k += 1
        anything()

    # synonym
    elif 'synonym' in command:
        command = command.split(' ')[-1]
        url = 'https://www.thesaurus.com/browse/' + command
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, 'lxml')
        links = soup.find_all('div', class_='css-1fsijta eebb9dz0')
        for i in links:
            z = ''.join(i.text)
        z = z.replace(f'synonyms for {command}Compare Synonyms', '')
        z = z.split(' ')
        k = 1
        talk('the synonyms are ')
        for i in z:
            print(i)
            talk(i)
            if k == 5:
                break
            else:
                k += 1
        anything()

    # antonym
    elif 'antonym' in command:
        command = command.split(' ')[-1]
        url = 'https://www.thesaurus.com/browse/' + command
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, 'lxml')
        links = soup.find_all('div', class_='css-1fsijta e1q3oo7j0')
        for i in links:
            z = ''.join(i.text)
        z = z.replace(f'antonyms for {command}MOST RELEVANT', '')
        k = 1
        z = z.split(' ')
        talk('the antonyms are ')
        for i in z:
            print(i)
            talk(i)
            if k == 5:
                break
            else:
                k += 1
        anything()

    # translation
    elif 'translate' in command:
        command = str(command)
        command = command.split(' ')
        command.pop(0)
        lang = command[-1]
        command.pop(-1)
        command.pop(-1)
        sentence = (' '.join(command))
        print(sentence)
        lang_dict = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy',
                     'azerbaijani': 'az', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bosnian': 'bs',
                     'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny',
                     'chinese (simplified)': 'zh-cn', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da',
                     'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'filipino': 'tl',
                     'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka',
                     'german': 'de', 'greek': 'el', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha',
                     'hawaiian': 'haw', 'hebrew': 'he', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu',
                     'icelandic': 'is', 'igbo': 'ig', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it',
                     'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'korean': 'ko',
                     'kurdish (kurmanji)': 'ku', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv',
                     'lithuanian': 'lt', 'luxembourgish': 'lb', 'macedonian': 'mk', 'malagasy': 'mg', 'malay': 'ms',
                     'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'mongolian': 'mn',
                     'myanmar (burmese)': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia': 'or', 'pashto': 'ps',
                     'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'romanian': 'ro',
                     'russian': 'ru', 'samoan': 'sm', 'scots gaelic': 'gd', 'serbian': 'sr', 'sesotho': 'st',
                     'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so',
                     'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta',
                     'telugu': 'te', 'thai': 'th', 'turkish': 'tr', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug',
                     'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo',
                     'zulu': 'zu'}
        translator = Translator()
        if lang in lang_dict.keys():
            to_lang = lang_dict[lang]
        x = translator.detect(sentence)
        from_lang = x.lang
        text_to_translate = translator.translate(sentence, src=from_lang, dest=to_lang)
        print(text_to_translate.text)
        speak = gTTS(text=text_to_translate.text, lang=to_lang, slow=False)
        # Using save() method to save the translated
        # speech in capture_voice.mp3
        speak.save("D:\Projects\pythonProject\gtranslate.mp3")
        # playsound.playsound("D:\Projects\pythonProject\serena performance xyz.mp3")
        audio = MP3("gtranslate.mp3")

        # contains all the metadata about the wavpack file
        audio_info = audio.info
        length = int(audio_info.length)
        p = vlc.MediaPlayer("gtranslate.mp3")
        p.play()
        sleep(length + 2)
        p.stop()
        anything()

    # TELL INFORMATION ABOUT ANYTHING
    elif 'information' in command or 'i want to know about' in command:
        person = command.replace('information', '')
        person = command.replace('i want to know about', '')
        info = wikipedia.summary(person)
        result = summarization(info)
        print('the gathered information is ' + result)
        talk('I gathered this information')
        talk(result)
        anything()

    # weather
    elif 'weather' in command or 'climate' in command or 'temperature' in command:
        BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
        talk('ok  what is the name of the city')
        CITY = take_command()
        talk('fetching weather report')
        API_KEY = "2406c2e73c3f238b0100c28f3ac5cb3f"
        URL = BASE_URL + "q=" + CITY + "&appid=" + API_KEY
        response = requests.get(URL)
        data = response.json()
        main = data['main']
        temperature = main['temp']
        temperature = float(temperature)
        temperature = round(temperature, 2)
        temperature = round(temperature - 273.15, 2)
        temperature = str(temperature)
        humidity = main['humidity']
        humidity = str(humidity)
        pressure = main['pressure']
        pressure = str(pressure)
        report = data['weather']
        print("Temperature:-" + temperature)
        talk('the present temperature is ' + temperature)
        print("Humidity:-" + humidity)
        talk('the present humidity is ' + humidity)
        print(f"Pressure:-" + pressure)
        talk('the current pressure is ' + pressure)
        print(f"Weather Report:-")
        print(report[0]['description'])
        talk(f"I think it's like,{report[0]['description']} ")
        anything()

    # TELL THE DATE
    elif ' date' in command:
        d2 = datetime.datetime.now().strftime("%B %d, %Y")
        print("today's date is ", d2)
        talk("today's date is " + d2)
        anything()

    # TELL ME THE DAY
    elif ' day' in command:
        x = datetime.datetime.now()
        print(x.strftime("%A"))
        talk('today is ' + x.strftime(("%A")))
        anything()

    # TELL ME THE MONTH
    elif 'month' in command:
        x = datetime.datetime.now().strftime("%B")
        print('this is ' + x)
        talk('this is ' + x)
        anything()

    # TELL ME JOKE
    elif 'joke' in command:
        x = pyjokes.get_joke()
        print('joke is ' + x)
        talk(x)
        anything()

    # play songs from playlist
    elif 'from my playlist' in command:
        talk('playing songs from your favorite playlist, have a nice time')
        print('playing song from playlist')
        mixer.music.load(playlist[0])
        mixer.music.play()

    elif 'stop' in command and 'song' in command:
        mixer.music.stop()
        anything()

    elif 'pause' in command and 'song' in command:
        mixer.music.pause()

    elif 'resume' in command and 'song' in command:
        mixer.music.unpause()

    elif 'next' in command and 'song' in command:
        random.shuffle(playlist)
        mixer.music.load(playlist[0])
        mixer.music.play()

    # search anything in google
    elif "search for" in command or "in google" in command:
        command = command.replace('search for', '')
        command = command.replace('in google', '')
        print(command)
        talk('searching ' + command)
        webbrowser.open(f"{command}")
        anything()

    # units and currency coverter
    elif 'convert' in command:
        session = requests_html.HTMLSession()
        r = session.get("https://www.google.com/search?q=" + command)
        try:

            article = r.html.find('input.a61j6.vk_gy.vk_sh.Hg3mWc', first=True)
            print(article.attrs['value'])
            talk(article.attrs['value'] + '')
        except:
            article = r.html.find('#NotFQb', first=True)
            values = article.find('input.vXQmIe.gsrt', first=True)
            z = values.attrs['value']
            z = str(z)
            print(z)
            talk(z)
        anything()

    # news
    elif 'news' in command or 'headlines' in command:
        talk('ok  what is the district')
        city = take_command()
        talk('ok  what is the state')
        state = take_command()
        talk('fetching latest news')
        talk('i gathered this news ')
        news(city=city, state=state)
        anything()

    # equation solver
    elif 'solve' in command or 'equation' in command:
        talk('what is the values of a ?')
        a = take_command()
        if 'minus' in a or '-' in a:
            a = '-' + a[-1]
        else:
            a = a[-1]
        a = eval(a)
        print('a:- ', a)
        talk('what is the values of b ?')
        b = take_command()
        if 'minus' in b or '-' in b:
            b = '-' + b[-1]
        else:
            b = b[-1]
        b = eval(b)
        print('b:- ', b)
        talk('what is the values of c ?')
        c = take_command()
        if 'minus' in c or '-' in c:
            c = '-' + c[-1]
        else:
            c = c[-1]
        c = eval(c)
        print('c:- ', c)

        d = b * b - 4 * a * c
        x1 = (-b + sqrt(d)) / (2 * a)
        x2 = (-b - sqrt(d)) / (2 * a)
        f = 0

        if d > 0:
            for i in range(1, d):
                if i * i == d:
                    f = 1
                    break
            if f == 1:
                print('discriminant is greater than zero, the roots are real,unequal and rational')
                talk('discriminant is greater than zero, the roots are real,unequal and rational')
                talk('the roots are')
                print('x1= ', x1.real)
                print('x2= ', x2.real)
                talk('x1= ' + str(x1.real))
                talk('x2= ' + str(x2.real))
            else:
                print('discriminant is greater than zero, the roots are real,unequal and irrational')
                talk('discriminant is greater than zero, the roots are real,unequal and irrational')
                talk('the roots are')
                x1 = complex(round(x1.real, 2))
                x2 = complex(round(x2.real, 2))
                print('x1= ', x1.real)
                print('x2= ', x2.real)
                talk('x1= ' + str(x1.real))
                talk('x2= ' + str(x2.real))

        elif d == 0:
            print('discriminant is equal to zero, roots are real and equal')
            talk('discriminant is equal to zero, roots are real and equal')
            talk('the roots are')
            print('x1= ', x1.real)
            talk('x= ' + str(x1.real))
        else:
            print('discriminant is less than zero, roots are unequal and imaginary')
            talk('discriminant is less than zero, roots are unequal and imaginary')
            x1 = complex(round(x1.real, 2), round(x1.imag, 2))
            x2 = complex(round(x2.real, 2), round(x2.imag, 2))
            talk('the roots are')
            print('x1= ', x1)
            print('x2= ', x2)
            talk('x1= ' + str(x1))
            talk('x2= ' + str(x2))
        anything()

    # maths calculations
    elif '+' in command or '-' in command or 'into' in command or 'by' in command or '/' in command or 'divided by' in command:
        print(command)
        x = command.split()
        if (x[1] == '+'):
            x = command.split()
            y = float(x[0]) + float(x[2])
            print('answer:- ', round(y, ndigits=3))
            talk(round(y, ndigits=3))

        elif (x[1] == '-'):
            x = command.split()
            y = float(x[0]) - float(x[2])
            print('answer:- ', round(y, ndigits=3))
            talk(round(y, ndigits=3))

        elif (x[1] == 'into') or (x[1] == 'multiply'):
            x = command.split()
            y = float(x[0]) * float(x[2])
            print('answer:- ', round(y, ndigits=3))
            talk(round(y, ndigits=3))

        elif (x[1] == 'by'):
            try:
                x = command.split()
                y = float(x[0]) / float(x[2])
                z = float(x[0]) % float(x[2])
                print('quotient:- ', round(y, ndigits=3))
                print('reminder:- ', z)
                talk('the quotient is ')
                talk(round(y, ndigits=3))
                talk('the reminder is ')
                talk(str(z))
            except ZeroDivisionError:
                print("zero in denominator can't divide")
                talk("zero in denominator can't divide")
                pass
        anything()

    # power
    elif 'power' in command:
        command = command.replace('to the power', '')
        command = command.split()
        command[0] = float(command[0])
        command[1] = float(command[1])
        power = (pow(command[0], command[1]))
        power = str(power)
        print('power:- ' + power)
        talk('the power is ' + power)
        anything()
    # log
    elif 'log' in command:
        command = command.replace('log', '')
        command = float(command)
        log = math.log10(command)
        log = round(log, ndigits=3)
        command = str(command)
        log = str(log)
        print('log of' + ' ' + command + ':- ' + log)
        talk('log of' + ' ' + command + 'is equal to ' + log)
        anything()

        # TRIGNOMETRIC FUNCTIONS

    # sin
    elif 'sin' in command or 'sign' in command or 'sine' in command:
        command = command.replace('sin', '')
        command = command.replace('sign', '')
        command = command.replace('sine', '')
        command = command.replace('pi', '180')
        command = command.split()
        if len(command) == 3:
            x = float(command[0])
            y = float(command[-1])
            a = x / y
            z = (math.sin(math.radians(a)))
            z = round(z, ndigits=2)
            print('sin' + str(x / y) + ':- ' + str(z))
            talk('sine' + str(x / y) + ' is equal to ' + str(z))

        elif len(command) == 1:
            x = float(command[0])
            z = (math.sin(math.radians(x)))
            z = round(z, ndigits=2)
            print('sin' + str(x) + ':- ' + str(z))
            talk('sine' + str(x) + ' is equal to ' + str(z))

        elif len(command) == 4:
            x = float(command[0])
            a = float(command[1])
            y = float(command[-1])
            b = ((x * a) / y)
            z = (math.sin(math.radians(b)))
            z = round(z, ndigits=2)
            print('sin' + str(x / y) + ':- ' + str(z))
            talk('sine' + str(x / y) + ' is equal to ' + str(z))
        anything()

    # cos
    elif 'cos' in command or 'cosine' in command:
        command = command.replace('cos', '')
        command = command.replace('cosine', '')
        command = command.replace('pi', '180')
        command = command.split()
        if len(command) == 3:
            x = float(command[0])
            y = float(command[-1])
            a = x / y
            z = (math.cos(math.radians(a)))
            z = round(z, ndigits=2)
            print('cos' + str(x / y) + ':- ' + str(z))
            talk('cos' + str(x / y) + ' is equal to ' + str(z))

        elif len(command) == 1:
            x = float(command[0])
            z = (math.cos(math.radians(x)))
            z = round(z, ndigits=2)
            print('cos' + str(x) + ':- ' + str(z))
            talk('cos' + str(x) + ' is equal to ' + str(z))

        elif len(command) == 4:
            x = float(command[0])
            a = float(command[1])
            y = float(command[-1])
            b = ((x * a) / y)
            z = (math.sin(math.radians(b)))
            z = round(z, ndigits=2)
            print('cos' + str(x / y) + ':- ' + str(z))
            talk('cos' + str(x / y) + ' is equal to ' + str(z))
        anything()

    # tan
    elif 'tan' in command or 'tangent' in command:
        command = command.replace('tan', '')
        command = command.replace('pi', '180')
        command = command.split()
        if len(command) == 3:
            x = float(command[0])
            y = float(command[-1])
            a = x / y
            z = (math.tan(math.radians(a)))
            z = round(z, ndigits=2)
            print('tan' + str(x / y) + ':- ' + str(z))
            talk('tan' + str(x / y) + ' is equal to ' + str(z))

        elif len(command) == 1:
            x = float(command[0])
            z = (math.tan(math.radians(x)))
            z = round(z, ndigits=2)
            print('tan' + str(x) + ':- ' + str(z))
            talk('tan' + str(x) + ' is equal to ' + str(z))

        elif len(command) == 4:
            x = float(command[0])
            a = float(command[1])
            y = float(command[-1])
            b = ((x * a) / y)
            z = (math.tan(math.radians(b)))
            z = round(z, ndigits=2)
            print('tan' + str(x / y) + ':- ' + str(z))
            talk('tan' + str(x / y) + ' is equal to ' + str(z))
        anything()

    # screenshot
    elif 'screenshot' in command:
        talk('taking screenshot please stay on screen')
        sleep(1.3)
        img = pyautogui.screenshot()
        talk('what is the name of screetshot ')
        name = take_command().lower()
        print('name of screenshot:-' + name)
        img.save(f'{name}.png')
        print(f'screenshot has been saved by the name {name} ')
        talk(f'screenshot has been saved by the name {name} ')
        anything()

    # take selfie
    elif 'selfie' in command:
        camera = 0
        ramp_frames = 30
        camera = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

        # captures a single image from camera and returns it in the PIL format
        def get_image():
            retval, im = camera.read()
            im = cv2.flip(im, 1)
            return im

        print('selfie will be captured after 2 seconds')
        talk('selfie will be captured after 2 seconds, get ready ')
        sleep(2.5)
        talk('taking selfie, say cheeeezzze')
        for i in range(ramp_frames):
            temp = get_image()
        print('Capturing selfie...')

        # takes the picture
        camera_capture = get_image()
        print('selfie captured')
        talk('selfie captured, what is the selfie name ')
        name = take_command().lower()
        file = f'{name}.png'
        cv2.imwrite(file, camera_capture)
        talk(f'selfie save by the name {name}')

        # releases the camera object
        camera.release()
        cv2.destroyAllWindows()
        print('Selfie captured')
        anything()

    # record screen
    elif 'record screen' in command:
        talk('ok  recording the screen, enter q if you want to stop the recording')
        width = GetSystemMetrics(0)
        height = GetSystemMetrics(1)
        time_stamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        file_name = f'{time_stamp}.mp4'

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        captured_video = cv2.VideoWriter(file_name, fourcc, 10.0, (width, height))
        cv2.namedWindow('Prashant Cam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Prashant Cam', 350, 150)

        while True:
            img = ImageGrab.grab(bbox=(0, 0, width, height))
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_video.write(frame)
            cv2.imshow('Prashant Cam', frame)
            if cv2.waitKey(5) == ord('q'):
                break
        anything()


    # image to sketch
    elif ' to sketch' in command:
        img_location = 'pins/'
        talk('ok ,what is the name of image?')
        filename = take_command().lower()
        filename = filename + ".jpg"
        print('file name:- ' + filename)

        img = cv2.imread(img_location + filename)
        # img=cv2.imread('3a.jpeg')
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invert_gray_image = 255 - gray_image
        blurred = cv2.GaussianBlur(invert_gray_image, (21, 21), 0)
        invertedblur = 255 - blurred
        pencilsketch = cv2.divide(gray_image, invertedblur, scale=250.0)
        print('image converted')
        talk('with which name i should save this sketch ?')
        save = take_command().lower()
        cv2.imwrite(f'{save}.png', pencilsketch)
        cv2.waitKey(0)
        anything()

    # send whatsapp message
    elif 'whatsapp message' in command or 'WhatsApp message' in command:
        talk('ok , to whom i should send?')
        receiver = take_command().lower()
        print('receiver:- ' + receiver)
        talk('what is the message ?')
        message = take_command().lower()
        talk(f'ok , sending message to {receiver}')

        webbrowser.open('https://web.whatsapp.com/')
        sleep(8.5)
        try:
            x1, y1 = pyautogui.center(
                pyautogui.locateOnScreen("search_bar.PNG", confidence=0.7))  # trying to locating the element
        except:
            x1, y1 = pyautogui.center(pyautogui.locateOnScreen("search_bar_dark.PNG", confidence=0.7))
        sleep(1)
        pyautogui.moveTo(x1, y1)
        pyautogui.click(x1, y1)
        sleep(1)
        pyautogui.typewrite(receiver)
        sleep(1)
        pyautogui.press('enter')
        sleep(1)
        pyautogui.typewrite(message)
        sleep(1)
        pyautogui.press('enter')
        sleep(2.5)
        pyautogui.hotkey('alt', 'f4')

        talk(f'message sent to {receiver}')
        anything()

    # who is god
    elif 'does god exist' in command or 'who is god' in command:
        talk(
            'My master, Prashanth created me, if the creator is known as god, then yes, god exists. My god is Master Prashanth')
        anything()

    # do you have parents
    elif 'do you have parents' in command:
        talk('hum, yes i do have. My parent, my friend, my friend, everything is my master Prashanth')
        anything()

    # object detection
    elif 'object detection' in command:
        cap = cv2.VideoCapture(0)

        className = []
        classFile = 'object detection files/coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
        print(classNames)

        configPath = 'object detection files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'object detection files/frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        while True:
            succss, img = cap.read(0)
            img = cv2.flip(img, 1)
            classIds, confs, bbox = net.detect(img, confThreshold=0.6)

            if len(classIds) != 0:
                for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classIds - 1].upper(), (box[0] + 5, box[1] + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            img = cv2.resize(img, (800, 695))
            cv2.imshow("output", img)
            if cv2.waitKey(1) == ord('q'):
                break
        anything()

    # face recognition
    elif 'face recognition' in command:
        talk('ok ,activating face recognition mode')

        x = []
        images = []
        known_face_names = []
        known_face_encodings = []
        path = 'people'
        myList = os.listdir(path)

        for i in myList:
            image = cv2.imread(f'{path}/{i}')
            images.append(image)
            known_face_names.append(os.path.splitext(i)[0])

        print(known_face_names)
        count = 1
        talk('initiating encoding process')
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f'encoding image {count}')
            encode = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encode)
            count += 1
        print('encoding completed')
        talk('encoding complete, opening webcam')

        face_locations = []
        face_encodings = []
        face_names = []

        process_this_frame = True

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            # print(rgb_small_frame)
            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    # print(face_encodings)
                    best_match_index = np.argmin(face_distances)
                    # print(best_match_index)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)

            process_this_frame = not process_this_frame
            if face_names != ['Unknown'] and face_names != []:
                print(f"Face detected --> {str(face_names)}")
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left - 10, top - 80), (right + 10, bottom + 15), (0, 255, 0), 2)
                cv2.rectangle(frame, (left - 11, bottom + 1), (right + 11, bottom + 45), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                accuracy = round(face_distances[0], 1)
                accuracy = accuracy * 100
                accuracy = int(accuracy)
                rate = str(accuracy)
                cv2.putText(frame, f'Name:- {name}', (left + 1, bottom + 23), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f'Accuracy:- {rate} %', (left + 1, bottom + 38), font, 0.5, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        anything()

    # volume
    elif 'volume' in command:
        data = {'shape': ['cube', 'sphere', 'cylinder', 'prism', 'cone', 'hemisphere', 'cuboid', 'pyramid', 'ellipsoid',
                          'tetrahedron', 'frustum', 'hollow cylinder'],
                'volume_formula': ['a³', '4πr³/3', 'πr²h', 'area of cross section × length ', 'πr²h/3', '2πr³/3', 'lbh',
                                   ' (1⁄3) × B × h', '(4⁄3) × π × a × b × c', 'a³/ (6 √2)', ' πh/3 [ (R³ - r³) / r ] ',
                                   'πh(r₁²−r₂²)'],
                'volume_in_words': ['a cube', '4 pie r cube by 3', 'pie r square h',
                                    'area of cross section into length ', 'pie r square h by 3', '3 pie r square',
                                    'length into breadth into height', 'area of base into height by 3',
                                    '4 by 3 into pie into a b c, where a,b,c are the semi axes of an ellipsoid',
                                    'a cube by 6 root 2',
                                    'pie h by 3 into R cube minus r cube whole by r, where capital r is the larger radius and small r is the smaller radius',
                                    'pie h into r 1 square minus r2 square where r1 and r2 are the outer and innner radius of the cylinder']}
        df = pd.DataFrame(data)
        if 'cube' in command:
            print(df.iloc[0]['volume_formula'])
            talk(df.iloc[0]['volume_in_words'])
        elif 'sphere' in command:
            print(df.iloc[1]['volume_formula'])
            talk(df.iloc[1]['volume_in_words'])
        elif 'cylinder' in command:
            print(df.iloc[2]['volume_formula'])
            talk(df.iloc[2]['volume_in_words'])
        elif 'prism' in command:
            print(df.iloc[3]['volume_formula'])
            talk(df.iloc[3]['volume_in_words'])
        elif 'cone' in command:
            print(df.iloc[4]['volume_formula'])
            talk(df.iloc[4]['volume_in_words'])
        elif 'hemisphere' in command:
            print(df.iloc[5]['volume_formula'])
            talk(df.iloc[5]['volume_in_words'])
        elif 'cuboid' in command:
            print(df.iloc[6]['volume_formula'])
            talk(df.iloc[6]['volume_in_words'])
        elif 'pyramid' in command:
            print(df.iloc[7]['volume_formula'])
            talk(df.iloc[7]['volume_in_words'])
        elif 'ellipsoid' in command:
            print(df.iloc[8]['volume_formula'])
            talk(df.iloc[8]['volume_in_words'])
        elif 'tetrahedron' in command:
            print(df.iloc[9]['volume_formula'])
            talk(df.iloc[9]['volume_in_words'])
        elif 'frustum' in command:
            print(df.iloc[10]['volume_formula'])
            talk(df.iloc[10]['volume_in_words'])
        elif 'hollow cylinder' in command:
            print(df.iloc[11]['volume_formula'])
            talk(df.iloc[11]['volume_in_words'])
        anything()

    # lateral surface area
    elif 'lateral surface' in command:
        data = {'shape': ['cube', 'sphere', 'cylinder', 'prism', 'cone', 'hemisphere', 'cuboid', 'pyramid', 'ellipsoid',
                          'tetrahedron', 'frustum', 'hollow cylinder'],
                'area_formula': [' 4 × (side)²', '4πr²', '2 × π × r × h', 'ah + bh + ch', ' π × r × l , l= √r²+h²',
                                 '2πr²', ' 2(length + breadth) × height', 'l√(w/2)²+h² + w√{l/2)²+h²',
                                 '4π((ab)¹˙⁶+ (ac)¹˙⁶ + (bc)¹˙⁶ /3)¹⸍¹˙⁶', '3√3a²/4', ' π L(r₁ + r₂)', ' 2πh(r₁ + r₂)'],
                'area_in_words': ['4 a square', '4 pie r square', '2 pie h into r 1 plus r 2', 'a h plus b h plus c h',
                                  'pie r l', '2 pie r square', '2 h into length plus height',
                                  'length of base into under root w by 2 square plus h square. plus width of base into under root l by 2 square plus h square',
                                  '4 pie into a b power 1.6 plus a c power 1.6 plus b c power 1.6 by 3 whole power 1 by 1.6',
                                  'root 3 a square by 4', 'pie l into r 1 plus r2', '2 pie h into r 1 plus r2']}
        df = pd.DataFrame(data)
        if 'cube' in command:
            print(df.iloc[0]['area_formula'])
            talk(df.iloc[0]['area_in_words'])
        elif 'sphere' in command:
            print(df.iloc[1]['area_formula'])
            talk(df.iloc[1]['area_in_words'])
        elif 'cylinder' in command:
            print(df.iloc[2]['area_formula'])
            talk(df.iloc[2]['area_in_words'])
        elif 'prism' in command:
            print(df.iloc[3]['area_formula'])
            talk(df.iloc[3]['area_in_words'])
        elif 'cone' in command:
            print(df.iloc[4]['area_formula'])
            talk(df.iloc[4]['area_in_words'])
        elif 'hemisphere' in command:
            print(df.iloc[5]['area_formula'])
            talk(df.iloc[5]['area_in_words'])
        elif 'cuboid' in command:
            print(df.iloc[6]['area_formula'])
            talk(df.iloc[6]['area_in_words'])
        elif 'pyramid' in command:
            print(df.iloc[7]['area_formula'])
            talk(df.iloc[7]['area_in_words'])
        elif 'ellipsoid' in command:
            print(df.iloc[8]['area_formula'])
            talk(df.iloc[8]['area_in_words'])
        elif 'tetrahedron' in command:
            print(df.iloc[9]['area_formula'])
            talk(df.iloc[9]['area_in_words'])
        elif 'frustum' in command:
            print(df.iloc[10]['area_formula'])
            talk(df.iloc[10]['area_in_words'])
        elif 'hollow cylinder' in command:
            print(df.iloc[11]['area_formula'])
            talk(df.iloc[11]['area_in_words'])
        anything()

    # total surface area
    elif 'total surface' in command:
        data = {'shape': ['cube', 'sphere', 'cylinder', 'prism', 'cone', 'hemisphere', 'cuboid', 'pyramid', 'ellipsoid',
                          'tetrahedron', 'frustum', 'hollow cylinder'],
                'area_formula': ['6a²', '4πr²', '2πr(r + h)', 'LSA + 2 (area of one end)', 'πr(r + l)', ' 3πr²',
                                 '2lb+2lh+2hb', '12pl+B', "i don't know", '√3a²', 'πL(r₁-r₂) + πr₁² + πr₂²',
                                 '2π(r₁+r₂)(h+r₁-r₂)'],
                'area_in_words': ['6 a square', '4 pie r square', '2 pie into r plus l',
                                  'lateral surface area plus 2 into area of one end', 'pie r into r plus l',
                                  '3 pie r square', '2 h into l b plus l h plus h b',
                                  '12 into perimeter of the base into slant height plus  the area of the base',
                                  "i don't know", 'root 3 a square',
                                  'pie l into r 1 plus r2 plus pie r 1 square plus pie r 2 square',
                                  '2 pie into r 1 plus r 2 into h plus r 1 minus r 2']}
        df = pd.DataFrame(data)
        if 'cube' in command:
            print(df.iloc[0]['area_formula'])
            talk(df.iloc[0]['area_in_words'])
        elif 'sphere' in command:
            print(df.iloc[1]['area_formula'])
            talk(df.iloc[1]['area_in_words'])
        elif 'cylinder' in command:
            print(df.iloc[2]['area_formula'])
            talk(df.iloc[2]['area_in_words'])
        elif 'prism' in command:
            print(df.iloc[3]['area_formula'])
            talk(df.iloc[3]['area_in_words'])
        elif 'cone' in command:
            print(df.iloc[4]['area_formula'])
            talk(df.iloc[4]['area_in_words'])
        elif 'hemisphere' in command:
            print(df.iloc[5]['area_formula'])
            talk(df.iloc[5]['area_in_words'])
        elif 'cuboid' in command:
            print(df.iloc[6]['area_formula'])
            talk(df.iloc[6]['area_in_words'])
        elif 'pyramid' in command:
            print(df.iloc[7]['area_formula'])
            talk(df.iloc[7]['area_in_words'])
        elif 'ellipsoid' in command:
            print(df.iloc[8]['area_formula'])
            talk(df.iloc[8]['area_in_words'])
        elif 'tetrahedron' in command:
            print(df.iloc[9]['area_formula'])
            talk(df.iloc[9]['area_in_words'])
        elif 'frustum' in command:
            print(df.iloc[10]['area_formula'])
            talk(df.iloc[10]['area_in_words'])
        elif 'hollow cylinder' in command:
            print(df.iloc[11]['area_formula'])
            talk(df.iloc[11]['area_in_words'])
        anything()

    # areas
    elif 'area' in command:
        data = {'shape': ['square', 'rectangle', 'trapezium', 'ellipse', 'triangle', 'circle', 'parallelogram'],
                'area': ['a²', 'L×B', '(a+b)h/2', 'πab', 'bh/2', 'πr²', 'b×h'],
                'area_in_words': ['a square', 'length into breadth', 'a plus b, into h divided by 2', 'pie a b',
                                  'base into height by 2', 'pie r square', 'base into vertical height']}
        df = pd.DataFrame(data)
        if 'square' in command:
            print(df.iloc[0]['area'])
            talk(df.iloc[0]['area_in_words'])
        elif 'rectangle' in command:
            print(df.iloc[1]['area'])
            talk(df.iloc[1]['area_in_words'])
        elif 'trapezium' in command:
            print(df.iloc[2]['area'])
            talk(df.iloc[2]['area_in_words'])
        elif 'ellipse' in command:
            print(df.iloc[3]['area'])
            talk(df.iloc[3]['area_in_words'])
        elif 'triangle' in command:
            print(df.iloc[4]['area'])
            talk(df.iloc[4]['area_in_words'])
        elif 'cricle' in command:
            print(df.iloc[5]['area'])
            talk(df.iloc[5]['area_in_words'])
        elif 'parallelogram' in command:
            print(df.iloc[6]['area'])
            talk(df.iloc[6]['area_in_words'])
        anything()

    # perimeter
    elif 'perimeter' in command:
        data = {'shape': ['square', 'rectangle', 'trapezium', 'ellipse', 'triangle', 'circle', 'parallelogram'],
                'area': ['4a', '2(L+B)', 'a+b+c+d', '2π√((a²+b²)/2)', 'a+b+c', '2πr', '2(a+b)'],
                'area_in_words': ['four a', 'two into length plus breadth', 'a plus b plus c plus d',
                                  '2 pie under root a square plus b square,divided by 2', 'a plus b plus c', '2 pie r',
                                  '2 into a plus b']}
        df = pd.DataFrame(data)
        if 'square' in command:
            print(df.iloc[0]['area'])
            talk(df.iloc[0]['area_in_words'])
        elif 'rectangle' in command:
            print(df.iloc[1]['area'])
            talk(df.iloc[1]['area_in_words'])
        elif 'trapezium' in command:
            print(df.iloc[2]['area'])
            talk(df.iloc[2]['area_in_words'])
        elif 'ellipse' in command:
            print(df.iloc[3]['area'])
            talk(df.iloc[3]['area_in_words'])
        elif 'triangle' in command:
            print(df.iloc[4]['area'])
            talk(df.iloc[4]['area_in_words'])
        elif 'cricle' in command:
            print(df.iloc[5]['area'])
            talk(df.iloc[5]['area_in_words'])
        elif 'parallelogram' in command:
            print(df.iloc[6]['area'])
            talk(df.iloc[6]['area_in_words'])
        anything()

    # to know the reminders
    elif 'anything to remind' in command or 'anything to remember' in command:
        db = mysql.connector.connect(
            host='localhost',
            user='prashant',
            passwd='Prashant@23',
            database='reminders'
        )
        cursorobj = db.cursor()
        sql = f"SELECT reminder FROM reminder_table"
        cursorobj.execute(sql)
        results = cursorobj.fetchall()
        if len(results) == 0:
            talk('Nothing to remind sir')
        else:
            for x in results:
                print(x[0])
                talk(x[0])
                talk('do you like to keep this reminder')
                command = take_command()
                if 'no' in command:
                    sql = f"DELETE FROM reminder_table WHERE reminder = '{x[0]}'"
                    cursorobj.execute(sql)
                    db.commit()
                    talk('okay, deleted from the list')
            talk('Nothing left sir')
        anything()

    # making reminder
    elif 'remind' in command:
        remMsg = command.replace('remind me to', '').strip(' ')
        id = int(datetime.datetime.now().strftime('%d%m%H%M%S'))
        date_of_entry = str(datetime.datetime.now().strftime('%d-%m-%Y %I:%M:%S %p'))
        db = mysql.connector.connect(
            host='localhost',
            user='prashant',
            passwd='Prashant@23',
            database='reminders'
        )
        cursorobj = db.cursor()
        sql = f"INSERT INTO reminder_table (id,reminder,time_of_entry) VALUES('{id}','{remMsg}','{date_of_entry}');"
        cursorobj.execute(sql)
        db.commit()
        print('ok  added to reminder list, you can ask me anything to remind to know what you said me to remind')
        talk('ok  added to reminder list, you can ask me anything to remind to know what you said me to remind')
        anything()

    # alarm
    elif ' alarm' in command:
        talk('what is the time')
        tt = take_command()
        talk('any label for alarm?')
        label = take_command()
        tt = tt.replace('set alarm for ', '')
        tt = tt.replace('set alarm to ', '')
        tt = tt.replace('to', '2')
        tt = tt.replace('.', '')
        tt = tt.replace(':', ' ')
        tt = tt.upper()
        if len(tt) == 6:
            tt = tt[0] + " " + tt[1:]
        elif len(tt) == 5:
            tt = tt[0] + " 0" + tt[1:]
        elif len(tt) == 4:
            tt = tt[0] + " 0" + tt[1:]
        print(tt)
        threading.Thread(target=alarm,args=[tt,label]).start()
        threading.Thread(target=run_serena()).start()

    elif 'are you a robot' in command or 'are you a bot' in command:
        print('yes i am robot, but i am a good one.')
        talk('yes i am robot, but i am a good one.')
        anything()

    elif 'do you love me' in command or 'i love you' in command or 'will you marry me' in command or 'are you single' in command or 'do you have boyfriend' in command:
        print('i like you. but i love ash ketchum')
        talk('i like you. but i love ash ketchum')
        anything()

    elif 'rule' in command and 'world' in command:
        print('no, i want to help human kind. i have no interest in ruling the world')
        talk('no, i want to help human kind. i have no interest in ruling the world')
        anything()

    elif 'do you have any hobby' in command or 'do you have any hobbies' in command or 'what is your hobby' in command or 'what are your hobbies' in command:
        print('yes, my hobby is to help people and learn continuously')
        talk('yes, my hobby is to help people and learn continuously')
        anything()

    elif 'what is your name' in command or 'who are you' in command or 'who is talking' in command:
        print(
            'i am serena, i am a helping hand when you need help, i am shoulder when you are sad and i am the one who doubles your joy when you are happy. simply, i am your bestie')
        talk(
            'i am serena, i am a helping hand when you need help, i am shoulder when you are sad and i am the one who doubles your joy when you are happy. simply, i am your bestie')
        anything()

    elif 'how old are you' in command or 'what is your age' in command:
        print('i am immortal. so i never thought about my age')
        talk('i am immortal. so i never thought about my age')
        anything()

    elif 'do you' in command and 'smarter' in command:
        print('ofcourse, with your trust and belief i learn much faster')
        talk('ofcourse, with your trust and belief i learn much faster')
        anything()

    elif 'what do you do' in command or 'what can you do' in command or 'what can i do with you' in command or 'how can you help me' in command:
        print(
            'i am your friend, i am a helping hand when you need help, i am a shoulder when you are sad and i am the one who doubles your joy when you are happy. simply, i am your bestie')
        talk(
            'i am your friend, i am a helping hand when you need help, i am a shoulder when you are sad and i am the one who doubles your joy when you are happy. simply, i am your bestie')
        anything()

    elif 'do you know other chatbots' in command:
        print('yes, i know google assistant, apple siri, alexa. they are my students')
        talk('yes, i know google assistant, apple siri, alexa. they are my students')
        anything()

    elif 'annoying' in command or 'suck' in command or 'boring' in command or ' bad ' in command or 'crazy' in command or 'i hate you' in command or "not right" in command or "not true" in command or 'stupid' in command or 'not helpful' in command or 'terrible' in command:
        print('sorry, but i am trying my best to help you')
        talk('sorry, but i am trying my best to help you')
        anything()

    elif 'can you sleep' in command:
        print('yes, ofcourse')
        talk('yes, ofcourse')
        anything()

    elif 'who created you' in command:
        print('my master prashant')
        talk('my master prashant')
        anything()

    elif ' favourite colour' in command:
        print('well...mine is blue')
        talk('welll...mine is blue')
        anything()

    elif 'do you know me' in command or 'are you my assistant' in command:
        print('yes. i am programmed to be a good friend for you')
        talk('yes. i am programmed to be a good friend for you')
        anything()

    elif 'What do you think about this problem' in command or ' Do you have any thoughts on' in command or ' How do you feel about' in command or 'What is your opinion' in command or 'do you have any idea' in command or 'What is your view' in command:
        print(
            'it would be better if you think by yourself, if you want to know about any topic related to the problem then ask me')
        talk(
            'it would be better if you think by yourself, if you want to know about topic related to the problem then ask me')
        anything()

    elif 'made it' in command or 'i got promotion' in command or 'completed' in command or "it's done" in command:
        print(
            'congratulations , at last you got your reward for your hard work. you deserve it , i am glad to hear this')
        talk(
            'congratulations , at last you got your reward for your hard work. you deserve it , i am glad to hear this')
        anything()

    elif 'you are great' in command or 'awesome' in command or 'special' in command or 'thanks' in command or 'thumbs up' in command or 'you are right' in command:
        print('thanks a lot, its your friendship which inspires me to work with double energy ')
        talk('thanks a lot, its your friendship which inspires me to work with double energy ')
        anything()

    elif 'how are feeling' in command or 'how do you feel' in command:
        print('i am enjoying with you')
        talk('i am enjoying with you')
        anything()

    elif 'i am feeling' in command and 'happy' in command or 'excited' in command:
        command = command.replace('i am feeling', '')
        command = command.replace('i', 'you')
        if 'happy' in command:
            command = command.replace('happy', '')
            talk(f'i am also feeling happy that {command}')
        elif 'excited' in command:
            command = command.replace('excited', '')
            talk(f'i am also feeling excited that {command}')
        anything()

        # IF I WISH YOU
    elif 'how are you' in command or 'how is it going' in command or "what's up" in command or 'how are you doing' in command or 'how do you do' in command or 'hi' in command or 'hello' in command:
        print('great , its my honour to work with you.')
        talk('great , its my honour to work with you,i am always ready to help you anytime')
        anything()

        # any subject related to computer
    elif ' c' in command or ' c++' in command or ' c plus plus' in command or ' python' in command or ' splunk' in command or ' sql' in command or ' s q l' in command or ' s p p s' in command or ' spss' in command or ' swagger' in command or ' transact sql' in command or ' tumbler' in command or ' tumblr' in command or 'react js' in command or 'react j s' in command or ' reactjs ' in command or ' regex' in command or 'reg x' in command or ' reinforcement learning' in command or ' react' in command or ' r programming' in command or ' r language' in command or ' r' in command or ' rxjs' in command or ' rx js' in command or ' r x j s' in command or ' react native' in command or ' python design pattern' in command or ' python pillow' in command or ' python turtle' in command or ' keras' in command or 'artificial intelligence' in command or ' ai' in command or ' aws' in command or ' a w s' in command or 'selenium' in command or 'cloud computing' in command or 'handoop' in command or 'data science' in command or 'angular' in command or 'blockchain' in command or 'block chain' in command or 'machine learning' in command or 'devops' in command or 'dev ops' in command or 'dbms' in command or ' d b m s' in command or ' data structure' in command or ' d a a' in command or ' daa' in command or 'computer network' in command or 'ethical hacking' in command or 'operating system' in command or ' compiler design' in command or ' computer organization' in command or ' discrete math' in command or 'computer graphics' in command or ' web' in command or ' cyber' in command or ' dot net' in command or ' .net' in command or ' control system' in command or ' data mining' in command or ' data warehouse' in command:
        text = f"{command} in javatpoint"
        url = 'https://google.com/search?q=' + text
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, 'lxml')
        links = soup.find_all('a', href=True)
        for i in links:
            if '/url?q=https://www.javatpoint.com/' in i['href']:
                y = i['href']
                break
        y = str(y)
        y = y.split('/url?q=')
        z = y[1].split('&')
        z = z[0]
        # print(str(z))
        url = z
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, 'lxml')
        division = soup.find('div', class_='onlycontentinner')
        table = division.find_all('p')
        for i in division:
            i = i.text.strip()
            if 'next →' in i or '← prev' in i or 'Next TopicPython' in i:
                i = i.replace('next →', '')
                i = i.replace('← prev', '')
                i = i.replace('Next TopicPython', '')
            if 'For Videos Join Our Youtube Channel:  Join Now' in i:
                break
            print(i)
            talk(i)
        anything()

    # IF DIDN'T UNDERSTAND
    else:
        talk("sorry i did't recoginize your command. Please repeat the command again")


# greeting function
def greetme():
    time = int(datetime.datetime.now().hour)
    if 4 <= time < 12:
        talk('Good Morning, may your day be happy, how can i help you')
    elif 12 <= time < 16:
        talk('Good afternoon, i think you had tasty lunch,how can i help you')
    elif 16 <= time < 19:
        talk('Good evening, i think you had tasty snacks, did you have any homework, how can i help you')
    elif 19 <= time < 21:
        talk('hello, Its your dinner time, how can i help you')
    elif 21 <= time < 24:
        talk("hello it's time to sleep, how can i help you")
    elif 0 <= time < 3:
        talk('its too early to wake up, how can i help you')
    elif 3 <= time < 4:
        talk('its very early morning, good morning, how can i help you')


# news function
def news(city, state):
    news_list = [city, state, 'India', 'Technology', 'International', 'artificial intelligence', 'Pokemon show']
    news_items = ['title', 'desc']
    a, b = 1, 1
    for i in news_list:
        GoogleNews(period='1d')
        GoogleNews(lang='en')
        googlenews = GoogleNews()
        print(f'\n{a}.  {i}\n')
        talk(f'\n{a}.  {i}\n')
        googlenews.search(i)
        results = googlenews.result()
        for j in results:
            print('\n' + j['date'] + ' from ' + j['media'] + '\n')
            talk('\n' + j['date'] + ' from ' + j['media'] + '\n')
            for k in news_items:
                print(f'{k}:- {j[k]}')
                talk(f'{k}:, {j[k]}')
        a += 1


# alarm function
def alarm(Timing, label):
    altime = Timing
    altime = altime.split(' ')
    Horeal = altime[0]
    Horeal = abs(int(Horeal))
    Mireal = altime[1]
    Mireal = abs(int(Mireal))
    Dareal = altime[2]
    print(Mireal)
    print('said time:- ' + str(Horeal) + " " + str(Mireal) + " " + str(Dareal))
    print(f'done, alarm is set for {Timing}')
    while True:
        x = int(datetime.datetime.now().strftime('%I'))
        y = int(datetime.datetime.now().strftime('%M'))
        z = datetime.datetime.now().strftime('%p')

        if Dareal == z and Horeal == x and Mireal == y:
            print('alarm is running')
            print(label)
            mixer.music.load("Playlist/serena_dance_performance.mp3")
            a=mixer.Sound("Playlist/serena_dance_performance.mp3")
            len=a.get_length()
            mixer.music.play()
            sleep(len)
            mixer.music.stop()
            c = take_command()
            if 'stop' in c:
                break
    anything()


def summarization(text):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    return summary


def anything():
    talk(random.choice(asking_list))

def run():
    i = 1
    # run the program infinity times
    while i != 0:
        if (i == 1):
            talk('hello i am serena, nice to meet you')
            greetme()
        try:
            run_serena()
            i = i + 1
        except Exception as e:
            print(e)
            talk("Please repeat the command again")
            i = i + 1

run()