import cv2
import time
import speech_related
def selfie_capture():
    camera=0
    ramp_frames=30
    camera=cv2.VideoCapture(camera,cv2.CAP_DSHOW)
    def get_image():
        retval, im = camera.read()
        im=cv2.flip(im,1)
        return im

    print('taking selfie, say cheeeezzze')
    speech_related.talk('taking selfie, say cheeeezzze')
    time.sleep(1.5)
    for i in range(ramp_frames):
        temp=get_image()
    camera_capture=get_image()
    file="selfie.png"
    cv2.imwrite(file,camera_capture)
    speech_related.talk("selfie saved with name selfie.png")
