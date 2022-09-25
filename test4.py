import face_recognition as fr
import numpy as np
import cv2
import os

faces_path = "/Users/school/Documents/School/AHCompSci/opencv/faces"

def get_face_encodings():
    faces_names = os.listdir(f"{faces_path}//known")
