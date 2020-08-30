#Import opencv(cv2), Numpy array(numpy), time, keyboard
import cv2
import time
import numpy as np
import keyboard

#Uncomment the below code to save the output. This will only save the invisible part
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#To read frames from the webcam
cap = cv2.VideoCapture(0)

# Put system on sleep for 3 seconds before the webcam starts
time.sleep(3)
count = 0
background = 0
key=0

# Read every frame from the webcam, until the camera is open
while (cap.isOpened()):

    #start the magical cloak when key 1 is pressed
    if keyboard.is_pressed('1'):
        key = 1
        cv2.destroyAllWindows()
        # Capture the background in range of 60 for the first time
        for i in range(60):
            ret, background = cap.read()
            background = np.flip(background, axis=1)

    if key == 1:

        # Refresh the background if the background is changed by pressing r
        if keyboard.is_pressed('r'):
            for i in range(60):
                ret, background = cap.read()
                background = np.flip(background, axis=1)

        ret, img = cap.read()
        if not ret:
            break
        count += 1
        img = np.flip(img, axis=1)

        # Convert the color code from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Generating masks to detect red color
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask1 = mask1 + mask2

        # Open and Dilate the mask image
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Create an inverted mask to segment out the red color from the frame
        mask2 = cv2.bitwise_not(mask1)

        # Segment the red color part out of the frame using bitwise and with the inverted mask
        res1 = cv2.bitwise_and(img, img, mask=mask2)

        # Create image showing static background frame pixels only for the masked region
        res2 = cv2.bitwise_and(background, background, mask=mask1)

        # Generating the final output and writing
        finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)

        #uncomment this to write an output file
        #out.write(finalOutput)

        cv2.imshow("magic", finalOutput)
        cv2.waitKey(1)

    else:
        key =0
        ret, img = cap.read()
        img = np.flip(img, axis = 1)
        cv2.imshow("normal", img)
        #out.write(img)
        cv2.waitKey(1)

    #to run the normal video
    if keyboard.is_pressed('0'):
        key = 0
        cv2.destroyAllWindows()
        ret, img = cap.read()
        img = np.flip(img, axis=1)
        cv2.imshow("normal", img)
        #out.write(img)
        cv2.waitKey(1)

    #exit the code
    if keyboard.is_pressed('c'):
        break


cap.release()
#to save output file
#out.release()
cv2.destroyAllWindows()