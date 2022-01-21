import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import os

# Initializing the Cascade classifier
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
#list of known family members for 
# Load a sample picture.
first_image = face_recognition.load_image_file("IMG20201110173423.jpg")
first_face_encoding = face_recognition.face_encodings(first_image)[0]


# Load a second sample picture.
second_image = face_recognition.load_image_file("photo_2021-09-05_21-11-21.jpg")
second_face_encoding = face_recognition.face_encodings(second_image)[0]

# Load third sample picture.
third_image = face_recognition.load_image_file("photo_2021-09-05_21-11-21.jpg")
third_face_encoding = face_recognition.face_encodings(third_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    first_face_encoding,
    second_face_encoding,
    third_face_encoding
]
known_face_names = [
    "Sanidhy Shrivastava",
    "Kashish Shah",
    "Rebika"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


    
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Initializing the pyttsx3 engine.
    engine=pyttsx3.init()
    sound=engine.getProperty("voices")
    engine.setProperty("voice", sound[1].id)
    rate=engine.getProperty("rate")
    engine.setProperty("rate", 150)
    

    

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_small_frame = small_frame[:, :, ::-1]
    # Modified and easy version.
    rgb_small_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Only process every other frame of video to save time
    
    if cv2.waitKey(1) & 0xFF==ord('p'):
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        #face_names = []
            for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown Person"
            
            

            #use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                    engine.say(f"{name} is at your door.")
                    engine.runAndWait()
            
                else:
                    engine.say(f"{name} is at door.")
                    engine.runAndWait()
                

                face_names.append(name)
        process_this_frame = not process_this_frame

           

    

    # Display the results ==========Modified===================
    faces = faceCascade.detectMultiScale(
        rgb_small_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

     # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # For displaying the name
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(frame,name, (w, w), font, 1.0, (255, 255, 255), 1)


    
        
        



    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
