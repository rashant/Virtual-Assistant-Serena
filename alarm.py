import datetime
from pygame import mixer
from time import sleep
import speech_related
mixer.init()


def alarm(Timing, label):
    altime = Timing
    altime = altime.split(' ')
    Horeal = altime[0]
    Horeal = abs(int(float(str(Horeal.strip()))))
    Mireal = altime[1]
    Mireal = abs(int(float(str(Mireal.strip()))))
    Dareal = altime[2]
    print(Mireal)
    print('said time:- ' + str(Horeal) + " " + str(Mireal) + " " + str(Dareal))
    print(f'done, alarm is set for {Timing}')
    speech_related.talk(f'alarm set for {Timing}')
    while True:
        x = int(datetime.datetime.now().strftime('%I'))
        y = int(datetime.datetime.now().strftime('%M'))
        z = datetime.datetime.now().strftime('%p')

        if Dareal == z and Horeal == x and Mireal == y:
            print('alarm is running')
            print(label)
            a = mixer.Sound("Playlist/serena dance performance.mp3")
            len = a.get_length()
            mixer.music.load("Playlist/serena dance performance.mp3")
            mixer.music.play()
            sleep(5)
            mixer.music.stop()
            c = 'stop'
            if c == 'stop':
                break


# tt = '10:06 PM'
#
# label = 'nothing'
# tt = tt.replace('set alarm for ', '')
# tt = tt.replace('set alarm to ', '')
# tt = tt.replace('to', '2')
# tt = tt.replace('.', '')
# tt = tt.replace(':', ' ')
# tt = tt.upper()
# if len(tt) == 6:
#     tt = tt[0] + " " + tt[1:]
# elif len(tt) == 5:
#     tt = tt[0] + " 0" + tt[1:]
# elif len(tt) == 4:
#     tt = tt[0] + " 0" + tt[1:]
# print(tt)
#
