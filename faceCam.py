# Using Desktop Camera
from Camera_init import *
import os
import shutil
import cv2
import cv2
import time

cTime = 0
pTime = 0
h = 480  # y co-ordinate
w = int(1.3334*h)


def end():
    cap.release()
    cv2.destroyAllWindows()
def get_img(n):
    # Create the images directory if it does not exist
    list_=["red","green","blue"]

    num = 0
    a = 1
    while a:
        # ret, depth_frame, depth_img, color_frame = cap.get_frame()
        ret, color_frame = cap.read()


        # depth_img will give use depth distance other t1o are used for obtaining depth and color frame
        # color_frame = cv2.resize(color_frame, (w, h))
        if not ret:
            print("return value is", ret)
            continue

        # cv2.imshow("color_frame", color_frame)
        cv2.waitKey(1) & 0xFF
        # if key_pressed == ord('q'):
        #     break

        cv2.imwrite('images/'+list_[n] + '.jpg', color_frame)
        print("image saved!")
        num += 1

        a = 0
        return 1





if os.path.exists('images'):
    shutil.rmtree('images')
os.makedirs('images')
cap = cv2.VideoCapture(0)



