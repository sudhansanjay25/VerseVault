import face_recognition
import os
import numpy as np

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")

def load_known_faces():
    known_encodings = []
    known_names = []

    for file_name in os.listdir(KNOWN_FACES_DIR):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(KNOWN_FACES_DIR, file_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file_name)[0])  # filename without extension

    return known_encodings, known_names

def match_face(new_encoding, known_encodings, known_names, tolerance=0.5):
    matches = face_recognition.compare_faces(known_encodings, new_encoding, tolerance)
    if True in matches:
        matched_index = matches.index(True)
        return known_names[matched_index]
    return None
