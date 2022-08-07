from speech_related import *
import bs4 as bs
import re
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

# print(to_search)
def google(to_search):
    to_search=to_search.replace(" ",'+')
    url='https://www.google.com/search?q='+to_search

    import requests
    from bs4 import BeautifulSoup
    page = requests.get("https://www.google.dz/search?q="+to_search)
    soup = BeautifulSoup(page.content,features="lxml")

    link_list =[]
    links = soup.findAll("a")
    for link in  soup.find_all("a",href=re.compile("(?<=/url\?q=)(htt.*://.*)")):
        link_list.append(re.split(":(?=http)",link["href"].replace("/url?q=","")))
    return link_list
def info(to_search):
    try:
            print("in javatpoint")
            link_list = google(to_search)
            javatpoint = 0
            java_url = ''
            linkss=''
            #extracting javatpoint link
            for i in link_list:
                if 'javatpoint' in i[0]:
                    javatpoint = 1
                    java_url=i[0]
                    break
            # Filtering URL
            for ch in java_url:
                if ch == '&':
                    break
                else:
                    linkss+=ch

            java_url = linkss
            req = Request(java_url)
            webpage = urlopen(req).read()
            soup=bs.BeautifulSoup(webpage,'html.parser')
            text = ' '
            stop_count=0
            page = requests.get(java_url)
            # print(page.status_code)
            soup = BeautifulSoup(page.text, "html.parser")

            for para in soup.find_all('p'):
                text+=para.text
                if '.' in text:
                    stop_count+=1
                if stop_count == 4:
                    break
            print(text)
            talk(text)

    except:
        try:
                print("in wiki")
                to_search+=' wiki'
                link_list = google(to_search)
                javatpoint = 0
                java_url = ''
                linkss=''
                #extracting javatpoint link
                for i in link_list:
                    if 'en.wikipedia' in i[0]:
                        javatpoint = 1
                        java_url=i[0]
                        break
                # Filtering URL
                for ch in java_url:
                    if ch == '&':
                        break
                    else:
                        linkss+=ch

                java_url = linkss
                req = Request(java_url)
                webpage = urlopen(req).read()
                soup=bs.BeautifulSoup(webpage,'html.parser')
                text = ''
                stop_count=0
                page = requests.get(java_url)
                # print(page.status_code)
                soup = BeautifulSoup(page.text, "html.parser")

                for para in soup.find_all('p'):
                    text+=para.text
                    if '.' in text:
                        stop_count+=1
                    if stop_count == 1:
                        break
                text= re.sub(r'\[[0-9]*]',' ',text)
                text = re.sub(r'\s+',' ',text)
                # text = text.lower()
                text = re.sub(r'\d',' ',text)
                text = re.sub(r'\s+',' ',text)
                print(text)
                talk(text)
        except:
            print('sorry, no information found')
            talk('sorry, no information found')
