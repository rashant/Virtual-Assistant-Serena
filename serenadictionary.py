import requests_html
session = requests_html.HTMLSession()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
def meaning(word):
    r = session.get("https://www.dictionary.com/browse/" + word)
    meaning = r.html.find('span.one-click-content.css-nnyc96.e1q3nk1v1')
    mean=[]
    for i in meaning:
        mean.append(i.text)
    return mean

def synonyms(word):
    r = session.get("https://www.thesaurus.com/browse/" + word)
    synonyms = r.html.find('a.css-1kg1yv8.eh475bn0')
    synonym=[]
    for i in synonyms:
        synonym.append(i.text)
    return synonym

def antonyms(word):
    r = session.get("https://www.thesaurus.com/browse/" + word)
    antonyms = r.html.find('a.css-15bafsg.eh475bn0')
    antonymsx = r.html.find('a.css-pc0050.eh475bn0')
    antonym=[]
    for i in antonyms:
        antonym.append(i.text)
    for i in antonymsx:
        antonym.append(i.text)
    return antonym

def use_this_word(word):
    r = session.get("https://www.thesaurus.com/browse/" + word)
    sentence = r.html.find('div.css-8xzjoe.e15rdun50')
    sentences=[]
    for i in sentence:
        sentences.append(i.text)
    return sentences

