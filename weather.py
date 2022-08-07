import requests_html
session = requests_html.HTMLSession()

def temperature(command='gwalior'):
    print(command)
    r = session.get("https://www.google.com/search?q="+" weather in "+command)
    temp = r.html.find('span.wob_t.q8U8x', first=True).text
    other_info= r.html.find('div.wtsRwe', first=True).text
    day_type= r.html.find('div.wob_dcp', first=True).text

    other_info=other_info.split('\n')
    dictionary={'temperature':temp,'Day Type':day_type}
    for i in other_info:
        x,y=i.split(':')
        if x=="Wind":
            y=y.split('h')[0]+'h'
        dictionary[x]=y
    return dictionary

