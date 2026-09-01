import face_recognition
import cv2
import numpy as np
import requests

# Load known encodings from DB (could be exposed via Django API)
known_faces = [...]  # Load with IDs and encodings

video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding in encodings:
        matches = face_recognition.compare_faces([f['encoding'] for f in known_faces], encoding, tolerance=0.5)
        name = "Unauthorized"
        if True in matches:
            index = matches.index(True)
            name = known_faces[index]['name']
            # Hit Django endpoint to mark as authorized visit
            requests.post("http://localhost:8000/api/face-log/", json={"name": name, "status": "authorized"})
        else:
            requests.post("http://localhost:8000/api/face-log/", json={"status": "unauthorized"})
