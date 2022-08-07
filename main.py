import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import alarm
import datetimemodule
import greetme
import localmusic
import news
import selfie
import serenadictionary
import serenatranslator
import serenajokes
import unitconverter
import weather
import webscrapper
from speech_related import *
from gtts import gTTS
import torchmetrics

# initialize metric
metric = torchmetrics.Accuracy()

stemmer = PorterStemmer()
import nltk


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


with open('intents.json', 'r') as f:
    intents = json.load(f)
all_words = []
tags = []
xy = []
for intent in intents["intents"]:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(all_words), 'unique stemmed words')
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, num_classes)
        # self.l6 = nn.Linear(hidden_size, hidden_size)
        # self.l7 = nn.Linear(hidden_size, hidden_size)
        # self.l8 = nn.Linear(hidden_size, hidden_size)
        # self.l9 = nn.Linear(hidden_size, hidden_size)
        # self.l10 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        # out = self.relu(out)
        # out = self.l6(out)
        # out = self.relu(out)
        # out = self.l7(out)
        # out = self.relu(out)
        # out = self.l8(out)
        # out = self.relu(out)
        # out = self.l9(out)
        # out = self.relu(out)
        # out = self.l10(out)
        # no activation and no softmax at the end
        return out


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


#
#
num_epochs = 10000
batch_size = 32
learning_rate = 0.0001
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)
print(input_size, output_size)
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#Train the model
#
# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
#         # Forward pass
#         outputs = model(words)
#         # if y would be one-hot, we must apply
#         # labels = torch.max(labels, 1)[1]
#         loss = criterion(outputs, labels)
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 100 == 0:
#         acc = metric(outputs,labels)
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}, Accuracy: {acc*100:.6f}')
# print(f'final loss: {loss.item():.6f}')
# acc = metric.compute()
# print(f"Accuracy on all data: {acc}")
#
# # Reseting internal state such that metric ready for new data
# metric.reset()
# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }
# File = "data.pth"
# torch.save(data, File)
#
# print("saved")
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Erica"
print("Let's chat! (type 'quit' to exit)")
initial = 0
while True:
        if initial == 0:
            print(greetme.greetme())
            talk(greetme.greetme())
        initial += 1
        # sentence = "do you use credit cards?"
        sentence = input("enter:- ")
        print("You said: ", sentence)

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
    #try:
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        print("tag:- ", tag)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print("intents['tag']:-  ", intent['tag'])
                    ans=random.choice(intent['responses'])
                    print(f"{bot_name}: {ans}")
                    talk(ans)
                    break

            for intent in intents['intents']:
                if tag == 'weather':
                    weather_report = weather.temperature()
                    print("temperature :- ", weather_report['temperature'])
                    print("humidity :- ", weather_report['Humidity'])
                    print("wind speed :- ", weather_report['Wind'])
                    print("Precipitation:- ", weather_report['Precipitation'])
                    print("Day type:- ", weather_report['Day Type'])
                    talk(f"the temperature is {weather_report['temperature']} degree celcius with precipitation of {weather_report['Precipitation']}, wind speed of {weather_report['Wind']} and humidity of {weather_report['Humidity']}. The day is most likely to be {weather_report['Day Type']}")
                    break

                elif tag == 'alarm':
                    print(alarm.alarm('10:00 PM', 'sleep'))
                    break

                elif tag == 'time':
                    timeprint,timespeak=datetimemodule.time()
                    print(timeprint)
                    talk(timespeak)
                    break

                elif tag == 'date':
                    print(datetimemodule.date())
                    talk(datetimemodule.date())
                    break

                elif tag == 'day':
                    print(datetimemodule.day())
                    talk(datetimemodule.day())
                    break

                elif tag == 'month':
                    print(datetimemodule.month())
                    talk(datetimemodule.month())
                    break

                elif tag == 'year':
                    print(datetimemodule.year())
                    talk(datetimemodule.year())
                    break

                elif tag == 'week':
                    print(datetimemodule.week())
                    talk(datetimemodule.week())
                    break

                elif tag == 'songs':
                    localmusic.music()
                    break

                elif tag == 'nextsong':
                    localmusic.next()
                    break

                elif tag == 'pausesong':
                    localmusic.pause()
                    break

                elif tag == 'resumesong':
                    localmusic.resume()
                    break

                elif tag == 'stopsong':
                    localmusic.stop()

                elif tag == 'news':
                    news.news()
                    break

                elif tag == 'selfie':
                    selfie.selfie_capture()
                    break

                elif tag == 'meaning':
                    all_word = sentence
                    ignore_word = ['?', '.', '!', 'what', 'meaning', 'of', 'is', 'the', 'mean', 'by', 'do', 'you']
                    all_word = [w for w in all_word if w not in ignore_word]
                    meaning = serenadictionary.meaning(all_word[-1])
                    count=0
                    for i in meaning:
                        print(i)
                        talk(i)
                        count+=1
                        if count==3:
                            break
                    break

                elif tag == 'synonym':
                    all_word = sentence
                    ignore_word = ['?', '.', '!', 'what', 'synonym', 'synonyms', 'of', 'is', 'the', 'are']
                    all_word = [w for w in all_word if w not in ignore_word]
                    synonym = serenadictionary.synonyms(all_word[-1])
                    count=0
                    for i in synonym:
                        print(i)
                        talk(i)
                        count+=1
                        if count==3:
                            break
                    break

                elif tag == 'antonym':
                    all_word = sentence
                    ignore_word = ['?', '.', '!', 'what', 'antonym', 'antonyms', 'of', 'is', 'the', 'are']
                    all_word = [w for w in all_word if w not in ignore_word]
                    antonym = serenadictionary.antonyms(all_word[-1])
                    count=0
                    for i in antonym:
                        print(i)
                        talk(i)
                        count+=1
                        if count==3:
                            break
                    break

                elif tag == 'makesentence':
                    all_word = sentence
                    ignore_word = ['?', '.', '!', 'how', 'to', 'use', 'using', 'make', 'your', 'own', 'sentence',
                                   'with', 'word', 'the', 'a', 'an']
                    all_word = [w for w in all_word if w not in ignore_word]
                    sentence = serenadictionary.use_this_word(all_word[0])
                    count=0
                    for i in sentence:
                        print(i)
                        talk(i)
                        count+=1
                        if count==3:
                            break
                    break

                elif tag == 'convert':
                    sentence = sentence
                    unit=unitconverter.converter(''.join(sentence))
                    print(unit)
                    talk(unit)
                    break

                elif tag == 'translate':
                    all_word = sentence
                    ignore_word = ['translate', 'translation', 'of', 'to']
                    all_word = [w for w in all_word if w not in ignore_word]
                    lang=all_word[-1]
                    all_word.pop()
                    z = ' '.join(all_word)
                    text_to_translate = serenatranslator.translate(z, lang)
                    speak = gTTS(text=text_to_translate)
                    speak.save("gtranslate.mp3")
                    from pygame import mixer
                    mixer.init()
                    mixer.music.load("gtranslate.mp3")
                    print(text_to_translate)
                    mixer.music.play()
                    break

                elif tag == 'information':
                    all_word = sentence
                    ignore_word = ['say', 'something', 'about', 'information', 'the', 'a', 'an', 'what', 'do', 'tell',
                                   'I', 'want', 'wanna', 'to', 'know', 'me', 'is']
                    all_word = [w for w in all_word if w not in ignore_word]
                    z = ' '.join(all_word)
                    webscrapper.info(z)
                    break

                elif tag=='joke':
                    print(serenajokes.joke())
                    talk(serenajokes.joke())
                    break

                elif tag=='bye':
                    exit()

        else:
            print(f"{bot_name}: I do not understand...")
            talk("I do not understand...")
        print("\n")
    #except Exception as e:
        #print("sorry there was a glitch, please repeat")
        #talk("sorry there was a glitch, please repeat")