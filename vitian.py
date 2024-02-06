from cvzone.HandTrackingModule import HandDetector #hand detection
import cv2  # ON the camera
import os  #file management
import numpy as np #use for adding multi dimensional array/list


# Parameters
width, height = 800, 600   # camera frames
gesturEThreshold = 300       #that green line
folderPath = "Presentation"    #the folder where we store our presentation

# Camera Setup
cap = cv2.VideoCapture(0)    #it is use for videocapture
cap.set(3, width )
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)   #1 is for detecting only one hand

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumberr = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
Startannotation = False
hs, ws = int(90* 3.5), int(6* 4)  # width and height of small image

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumberr])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    handss, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gesturEThreshold), (width, gesturEThreshold), (0, 255, 0), 10)

    if handss and buttonPressed is False:  # If hand is detected

        handd = handss[0]
        cx, cy = handd["center"]
        lmList = handd["lmList"]  # List of 21 Landmark points
        fingerss = detectorHand.fingersUp(handd)  # List of which fingers are up
        ForeFinger = lmList[8][0], lmList[8][1]
        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        ForeFinger = xVal, yVal

        if cy <= gesturEThreshold:  # If hand is at the height of the face
            # Gesture 1=left
            if fingerss == [1, 0, 0, 0, 0]: #if our hand is above this line and ouprint("left") r thumb is open then it print left
                print("Left")
                buttonPressed = True
                if imgNumberr > 0:
                    imgNumberr -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    Startannotation = False
                    # Gesture 2
            if fingerss == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumberr < len(pathImages) - 1:
                    imgNumberr += 1
                    annotations = [[]]
                    annotationNumber = -1
                    Startannotation = False
        # Gesture 3
        if fingerss == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, ForeFinger, 12, (0, 0, 255), cv2.FILLED)
        # Gesture 4
        if fingerss == [0, 1, 0, 0, 0]:
            if Startannotation is False:
                Startannotation = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(ForeFinger)
            cv2.circle(imgCurrent, ForeFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            Startannotation = False
        # Gesture 5
        if fingerss == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        Startannotation = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('e'):
        break