# Internal mediapipe module - hands was changed for detecting maximum no. of hands
# ctrl+Hands will get access to file, remove reader mode & set no. of points accordingly, line no. - 33
# ****************************************************************************************************************
import cv2
import pandas as pd
from Camera_init import *
import mediapipe as mp
import numpy as np
import random
import time


# Global Variable
point = (400, 300)
distance = 0
distList_max = [0, 0, 0, 0, 0]
distList_min = [0, 0, 0, 0, 0]

# Define the frame width and height - (650, 866) frame shape
w = int(1.3333*650)  # x co-ordinate
h = 650  # y co-ordinate


# Functions
def centroid(lmlist):
    cen_x = int((lmlist[5][1] + lmlist[9][1] + lmlist[13][1] + lmlist[17][1] + lmlist[0][1]) / 5)
    cen_y = int((lmlist[5][2] + lmlist[9][2] + lmlist[13][2] + lmlist[17][2] + lmlist[0][2]) / 5)
    xx, yy = cen_x, cen_y
    return xx, yy

def L_BwFing(pt1, pt2):
    return ((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)**0.5

def pointing_finger(lmList):
    global point, distList_max, distList_min
    xx, yy = centroid(lmList)
    

    # Finger condition
    for id in [1, 5, 9, 13, 17]:
        if distList_min[id//4] < L_BwFing(lmList[id], lmList[id+2]):
            distList_min[id//4] = L_BwFing(lmList[id], lmList[id+2])

    finger_pointing = []
    for id in [1, 5, 9, 13, 17]:
        # distList_max[i//4] >= L_BwFing(lmList[i],lmList[i+3])
        if L_BwFing(lmList[id],lmList[id+3]) > distList_min[id//4]:
            finger_pointing.append(id)

    if len(finger_pointing) != 0  and  len(finger_pointing)<3:
        random_num = random.choice(finger_pointing)
        xx = lmList[random_num+3][1]
        yy = lmList[random_num+3][2]

    # xx = lmList[8][1]
    # yy = lmList[8][2]
    if xx > w:
        xx = w-1
    if yy > h:
        yy = h-1
    point = (xx, yy)
    print(xx, yy)


# Initialize Intel Camera Realsense
dc = DepthCamera()
# 480, 640 frame shape

# Create Object for Hand
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Image Processing
pTime = 0
cTime = 0
Hand_data = []
index_data = []
while True:
    ret, depth_frame, depth_img, color_frame = dc.get_frame()
    # depth_img will give use depth distance other t1o are used for obtaining depth and color fram
    depth_img = cv2.resize(depth_img, (w, h))
    depth_frame = cv2.resize(depth_frame, (w, h))
    color_frame = cv2.resize(color_frame, (w, h))
    # color_frame = cv2.flip(color_frame, 1)
    # depth_frame = cv2.flip(depth_frame, 1)
    if ret == False:
        continue


    # Hand detection package from OpenCV
    imgRGB = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB) # Because mpHands module uses RGB
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    ind_x = 0
    ind_y = 0
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            # Draw hand landmarks
            mpDraw.draw_landmarks(color_frame, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = color_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if cx > w:
                    cx = w
                if cy > h:
                    cy = h
                # print(id, cx, cy)
                if id == 8:
                    ind_x = cx
                    ind_y = cy

            # Collect data
            if len(Hand_data) == 0:
                Hand_data = lmList
            else:
                Hand_data = np.vstack((Hand_data, lmList))


        # Find finger which is open
        print("Hand Img no.  = ", len(Hand_data) // 21)
        # print(lmList)
        pointing_finger(lmList)
        # cv2.circle(color_frame, (point[0], point[1]), 10, (0, 0, 255), cv2.FILLED)


    # fps at a point
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Show distance for a specific point
    distance = depth_img[point[1], point[0]]  # 1st y point, then x-point
    for i in range(300, 500):
        for j in range(300, 400):
            distt = depth_img[j, i]
            cv2.circle(color_frame, (i,j), 10, (0, 0, 255), cv2.FILLED)
            index_data.append(distt)
    # distance1 = depth_img[ind_y, ind_x]  # 1st y point, then x-point
    # index_data.append([ind_x, ind_y, distance1])
    # print(distance)



    # Display on screen
    String = str(int(fps)) + " & " + "{}mm".format(distance)
    cv2.putText(color_frame, String, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 180), 2)

    cv2.imshow("depth_frame", depth_frame)
    cv2.imshow("color_frame", color_frame)
    key = cv2.waitKey(1)
    if key == 110:  # chr(110) = n
        break

cv2.destroyWindow()
cv2.destroyAllWindows()
# print("\n\nList of maximum distance b/w fingers lm13:-\n", distList_min)
# print("Size of data: ", Hand_data.shape)
index_data = np.array(index_data)
df = pd.DataFrame(index_data, columns = ['cz'])
df.to_csv('index_data.csv', index=False, float_format='%.2f')
# print("And the hand lm data:-\n", Hand_data)
# # print(type(Hand_data))
# df = pd.DataFrame(Hand_data, columns=['id', 'cx', 'cy'])
# # print(type(df))
# df.to_csv('mydata.csv', index=False, float_format='%.2f')

