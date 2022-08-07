import datetime
def greetme():
    time = int(datetime.datetime.now().hour)
    if 4 <= time < 12:
        return('Good Morning, may your day be happy, how can i help you')
    elif 12 <= time < 16:
        return('Good afternoon, i think you had tasty lunch,how can i help you')
    elif 16 <= time < 19:
        return('Good evening, i think you had tasty snacks, did you have any homework, how can i help you')
    elif 19 <= time < 21:
        return('hello, Its your dinner time, how can i help you')
    elif 21 <= time < 24:
        return("hello, it's time to sleep, how can i help you")
    elif 0 <= time < 3:
        return('its too early to wake up, how can i help you')
    elif 3 <= time < 4:
        return('its very early morning, good morning, how can i help you')