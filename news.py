from GoogleNews import GoogleNews
import speech_related
def news():
    news_list = ['gwalior','madhya pradesh', 'India', 'Technology', 'International', 'artificial intelligence', 'Pokemon show']
    news_items = ['title', 'desc']
    a, b = 1, 1
    for i in news_list:
        GoogleNews(period='1d')
        GoogleNews(lang='en')
        googlenews = GoogleNews()
        print(f'\n{a}.  {i}\n')
        speech_related.talk(f'{a}.  {i}')
        googlenews.search(i)
        results = googlenews.result()
        for j in results:
            print('\n' + j['date'] + ' from ' + j['media'] + '\n')
            speech_related.talk(j['date'] + ' from ' + j['media'])
            for k in news_items:
                print(f'{k}:- {j[k]}')
                speech_related.talk(f'{k}:- {j[k]}')
        a += 1

