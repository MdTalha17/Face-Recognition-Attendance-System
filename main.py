import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("Resources/background.png")


#Importing the mode images
foldermodepath = "Resources/Modes"
modePathList = os.listdir(foldermodepath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(foldermodepath, path)))
    
 
#Load the Encoding file
print("Loading Encode File.... ")
with open('EncodeFile.p', 'rb') as file:   
    encodeListKnownwithIDs = pickle.load(file)
encodeListKnown, studentIDs = encodeListKnownwithIDs
print("Encode File Loaded")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture from webcam.")
        break
    
    #Resize and convert the frame for face detection
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    #Detect faces and encodings in the current frame
    FaceCurrFrame = face_recognition.face_locations(imgS)
    EncodeCurrFrame = face_recognition.face_encodings(imgS, FaceCurrFrame)
    
    #Overlay webcam feed on the  background image
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]
    
    #Process each detected face
    for EncodeFace, FaceLoc in zip(EncodeCurrFrame, FaceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, EncodeFace)
        FaceDist = face_recognition.face_distance(encodeListKnown, EncodeFace)

        matchIndex = np.argmin(FaceDist)

        if matches[matchIndex]:
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
    
    #Display the Images
    cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendence", imgBackground)
    
    #Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    