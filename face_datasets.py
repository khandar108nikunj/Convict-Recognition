import cv2
import os
import sqlite3


def assure_path_exists(path):
    dir = os.path.dirname ( path )
    if not os.path.exists ( dir ):
        os.makedirs ( dir )


vid_cam = cv2.VideoCapture ( 0 )

face_detector = cv2.CascadeClassifier ( 'haarcascade_frontalface_default.xml' )


def insertOrUpdate(Id, Name):
    conn = sqlite3.connect ( "FaceBase.db" )
    cmd = "SELECT * FROM People WHERE ID=" + str ( Id )
    cursor = conn.execute ( cmd )
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if (isRecordExist == 1):
        cmd = "UPDATE People SET Name=" + str ( Name ) + "WHERE ID=" + str ( Id )

    else:
        cmd = "INSERT INTO People(ID,Name) Values(" + str ( Id ) + ",'" + str ( Name ) + "')"

    conn.execute ( cmd )

    conn.commit ()
    conn.close ()


Id = input ( 'Enter User Id: ' )
Name = input ( 'Enter User Name: ' )
insertOrUpdate ( Id, Name )

count = 0

assure_path_exists ( "dataset/" )

while (True):

    _, image_frame = vid_cam.read ()

    gray = cv2.cvtColor ( image_frame, cv2.COLOR_BGR2GRAY )

    faces = face_detector.detectMultiScale ( gray, 1.3, 5 )

    for (x, y, w, h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle ( image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2 )

        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite ( "dataset/User." + str ( Id ) + '.' + str ( count ) + ".jpg", gray[y:y + h, x:x + w] )

        cv2.imshow ( 'frame', image_frame )

    if cv2.waitKey ( 100 ) & 0xFF == ord ( 'q' ):
        break

    elif count > 20:
        break

# Stop video
vid_cam.release ()

# Close all started windows
cv2.destroyAllWindows ()
