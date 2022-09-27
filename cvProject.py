import face_recognition as fr
from pytesseract import pytesseract
import numpy as np
import cv2
import os

from PIL import Image, ImageDraw
import face_recognition


# path to the folder with all the images containing faces to idenitfy - name of file = face name
faces_path = "/Users/school/Documents/School/AHCompSci/opencv/faces"

def get_face_encodings():
    face_names = os.listdir(f"{faces_path}")
    face_encodings = []
    face_landmarks_list = []

    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_path}//{name}")
        face_encodings.append(fr.face_encodings(face)[0])
        face_names[i] = name.split(".")[0]

        face_landmarks = face_recognition.face_landmarks(face)
        face_landmarks_list.append(face_landmarks)

    return face_encodings, face_names, face_landmarks_list



def find_Faces():

    face_encodings, face_names, face_landmarks_list = get_face_encodings()

    # store video from latpop camera in variable video
    video = cv2.VideoCapture(0)

    # scale factor of video image
    scl = 2

    while True:

        success, image = video.read()

        resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))

        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_image)
        unknown_encodings = fr.face_encodings(rgb_image, face_locations)

#-------------------------------------








#-------------------------------------

        for face_encoding, face_location in zip(unknown_encodings, face_locations):

            result = fr.compare_faces(face_encodings, face_encoding, 0.45)

            # draw box around and label the face if identified
            if True in result:
                name = face_names[result.index(True)]

                top, right, bottom, left = face_location

                cv2.rectangle(image, (left*scl, top*scl), (right*scl, bottom*scl), (0, 0, 255), 2)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left*scl, bottom*scl + 20), font, 0.8, (255, 255, 255), 1)


        # start test ------------------------------------------------------------------------

        face_landmarks_list = face_recognition.face_landmarks(image)

        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

        # Create a PIL imagedraw object so we can draw on the picture
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)


        for face_landmarks in face_landmarks_list:

            # Print the location of each facial feature in this image
            for facial_feature in face_landmarks.keys():
                print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

            # Let's trace out each facial feature in the image with a line!
            for facial_feature in face_landmarks.keys():
                d.line(face_landmarks[facial_feature], width=5)

        # Show the picture
        pil_image.show()

        # end test ------------------------------------------------------------------------

        cv2.imshow("frame", image)
        cv2.waitKey(1)



def find_Words(file):

    # get a pytesseract executable
    pytesseract.tesseract_cmd = "/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract"

    img = cv2.imread(file)

    # list of all words from the image returned from the pytesseract executable
    words_in_image = pytesseract.image_to_string(img)

    print("\n******************************")
    print("\n" + words_in_image + "\n")
    print("******************************")

    fileNameParts = file.partition(".")
    filename = fileNameParts[0] + ".txt" # fix filename

    # store words in a text file
    f = open(filename, "w")
    for line in words_in_image:
        f.write(line)
    f.close()


def test(name):

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("faces/" + name + ".jpg")

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)

    # Show the picture
    pil_image.show()

def main():

    print("\nWelcome to Hewitt's Open CV Project!!!")
    #file = input("\nPlease input the filename of an image: ")

    # Use tesseract to get words from an image store into text file
    #find_Words(file) # give name of image
    print("\nGet your files to find a text file containing all text from the image or look at a preview of it in terminal.") # fix when this prints

    name = input("Please input your name (Firstname_Lastname): ")
    test(name)

    print("\nNow, let's try to figure out who you are.")
    consent = input("Can I figure out who you are? (y/n): ")
    if consent == "y":

        # Use face_recognition library to identify faces using the laptop camera
        find_Faces()

main()

# to Do's

# comment
# fix keyboard interrupt at the end to end the video
# add readme that explains how this meets the grading requirements
# ask dr. j how enumerate works *************************************************

# finish comments and read me

# have the data - now just need to loop over the data and draw it on the image
