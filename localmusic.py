from pygame import mixer
import os
import random
import time

mixer.init()
music_dir = 'Playlist'
song = os.listdir(music_dir)
playlist=[]
for i in song:
    playlist.append('Playlist'+'/'+i)

def music():
    print(playlist[0])
    mixer.music.load(playlist[0])
    mixer.music.play()
    next()

def next():
    random.shuffle(playlist)
    mixer.music.load(playlist[0])
    mixer.music.play()

def stop():
    mixer.music.stop()

def pause():
    mixer.music.pause()

def resume():
    mixer.music.unpause()
