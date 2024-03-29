import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0) # here 0 corresponds to primary webcam connect to our device
# if we say 1 it means it is external webcam

fourcc=cv2.VideoWriter_fourcc(*'XVID')
# for saving video file which contains the feed from the webcam
#here .avi is the extension for the video
out = cv2.VideoWriter('invisibility cloak.avi', fourcc, 20.0, (640, 480))
# after executing the out statement the webcam is not stable immediately so if we give sleep for 2 sec it will become stable.
time.sleep(2)

#for capturing background
background = 0
# for creating a stable background
for i in range(30):
    ret, background = cap.read()  # here it captures the background and store the information or the image in the background variable
#here the cap.read is for reads the background

# the above for loop is for capturing the background. then while loop is for after capturing the background the person holding the cloak appears. so we need to perform some operations
while(cap.isOpened()):    # until the webcam keeps on capturing the while loop keeps on running.It will constantly keep looking for the cloak and as the moment it finds the cloak it will substitute the cloak part with the background
    ret, img = cap.read()  # this reads the img, i.e., person holding the cloak

    if not ret:
        break

#HSV - Hue saturation value
# Hue refers to the color, saturation refers to the darkness & lightness of the color, value refers to the brightness

# webcam directly captures in bgr color , so we need to convert it into hsv color
# this is because operating on hsv is easier compared to operating on bgr

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #hsv values
    # the below lines represents that we are instructing the program compiler or interpreter ihe program should consider the red color that ranges from edge value 0 to edge value 10
    #part 1 for masking - consider the background and ignore the cloak part
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
# now, we will create 2 masks - 1st will differentiate the background wrt cloak 2nd - will differentiate the cloak part wrt background
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    #part 2 for masking - consider the cloak part and ignore the background
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1  = mask1 + mask2  # bitwise_or
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 2)   # MORPH_OPEN - to remove the noise
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations = 1)  #MORPH_DILATE - smoothen the image or the feed

    mask2=cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background, background, mask = mask1) # used to differentiate the cloak color from the background
    res2 = cv2.bitwise_and(img, img, mask = mask2)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0) # it is basically like a linear equation so it is like (alpha * res1 + beta *res2 + gamma)

    cv2.imshow('Cloak project', final_output)
    k = cv2.waitKey(10)
    if k==27: # it is esc key have an asci value of 27
        break


cap.release()
cv2.destroyAllWindows()






