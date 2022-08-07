import cv2

colorset = "MWN$@%#&B89EGA6mK5HRkbYT43V0JL7gpaseyxznocv?jIftr1li*=-~^`':;,. "

imgpath = input("Path:")
img = cv2.imread(imgpath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
output = ""

for gray2 in gray:
    output += "\n"
    for dark in gray2:
        output += colorset[dark // 4] * 2

with open("output.txt", mode="w") as f:
    f.write(output)