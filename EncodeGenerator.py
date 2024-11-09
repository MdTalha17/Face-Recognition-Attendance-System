import cv2
import face_recognition
import os
import pickle

#Importing the student images
folderpath = "Images"
PathList = os.listdir(folderpath)
imgList = []
studentIDs = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderpath, path)))
    studentIDs.append(os.path.splitext(path)[0])
    
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList

print("Encoding started....")
encodeListKnown = findEncodings(imgList)
encodeListKnownwithIDs = [encodeListKnown, studentIDs]
print("Encoding Complete")   

file = open("EncodeFile.p", 'wb') 
pickle.dump(encodeListKnownwithIDs, file)
file.close()
print("File Saved")