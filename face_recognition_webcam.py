import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') # 'haar_face.xml'

people = ['Donald Trump', "Joe Biden"]
#features = np.load('features.npy', allow_pickle = True)
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(1)

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        #print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame,str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness = 2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
        if label == 0: # probably also need to set a limit on confidence
            cv.putText(frame,'OPEN THE DOOR FOR TRUMP', (20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(255,0,0), thickness = 5)

    cv.imshow("Video", frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

"""
img = cv.imread(r'C:\Stuff\Work\Programming\Python Stuff\Learning\Test\faces\Joe Biden\joe-biden.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

#detect the face
faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0),thickness = 2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness = 2)

cv.imshow("Detected Face",img)
cv.waitKey(0)
"""