import datetime
def date():
    now = datetime.datetime.now()
    return now.strftime("%B %d, %Y")

def time():
    now = datetime.datetime.now()
    hours = int(now.strftime("%I"))
    minutes = int(now.strftime("%M"))
    seconds = int(now.strftime("%S"))
    type=now.strftime("%p")
    return (now.strftime("%I:%M %S %p"),f"{hours} {minutes} {seconds} {type}")

def day():
    now = datetime.datetime.now()
    return now.strftime("%A")

def month():
    now = datetime.datetime.now()
    return now.strftime("%B")

def year():
    now = datetime.datetime.now()
    return now.strftime("%Y")

def week():
    now = datetime.datetime.now()
    return now.strftime("%U")