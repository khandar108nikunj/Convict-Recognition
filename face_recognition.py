import cv2
import sqlite3
import os


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX


def getProfile(Id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
        conn.close()
        return profile


cam = cv2.VideoCapture(0)

while True:
    ret, im = cam.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Get all face from frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # For each face in faces
    for (x, y, w, h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 10), (0, 255, 0), 4)

        # Recognize the face belongs to which ID
        
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            profile = getProfile(Id)
            if profile != None:
                cv2.putText(im, "Name : {0}".format(str(profile[1])), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0));
                cv2.putText(im, "Age : {0}".format(str(profile[2])), (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0));
                cv2.putText(im, "Gender : {0}".format(str(profile[3])), (x, y + h + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0));
                cv2.putText(im, 'Criminal Records : ' + str(profile[4]), (x, y + h + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255));

            # cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            # cv2.putText(im, str(profile[1]), (x,y-40), font, 1, (255,255,255), 3)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
            cv2.putText(im, str(id), (x+5,y-5), font, 1, (0,255,0), 2)
           
               
                
    cv2.imshow('im', im)

    if cv2.waitKey(1) == ord('q'):
                break
cam.release()
cv2.destroyAllWindows()
