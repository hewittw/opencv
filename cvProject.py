import face_recognition as fr
from pytesseract import pytesseract
import numpy as np
import cv2
import os
import face_recognition


# path to the folder with all the images containing faces to idenitfy - name of file = face name
faces_path = "/Users/school/Documents/School/AHCompSci/opencv/faces"

def get_face_encodings():

    # get names of faces and store in a list
    face_names = os.listdir(f"{faces_path}")

    # empty list to store the face data in
    face_encodings = []

    # go through each face and check against faces from images
    for i, name in enumerate(face_names):

        # use the facial recognition library to find faces images
        face = fr.load_image_file(f"{faces_path}//{name}")
        face_encodings.append(fr.face_encodings(face)[0])

        # change file name to face name to display later
        face_names[i] = name.split(".")[0]

    return face_encodings, face_names



def find_Faces():

    # get face info from intial face images and correspoinding names of each face
    face_encodings, face_names = get_face_encodings()

    # store video from latpop camera in variable video
    video = cv2.VideoCapture(0)

    # scale factor of video image
    scl = 2

    while True:

        # get screenshot of sorts
        success, image = video.read()

        # resize image
        resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))

        # convert to rgb for face_recognition library
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # get face data from faces on camera (program users)
        face_locations = fr.face_locations(rgb_image)
        unknown_encodings = fr.face_encodings(rgb_image, face_locations)

        # go through each face provided in folder and compare to faces on screen to 'find faces present on screen'
        for face_encoding, face_location in zip(unknown_encodings, face_locations):

            # see if the faces are the same based on the 'magical returned data'
            result = fr.compare_faces(face_encodings, face_encoding, 0.45)

            # draw box around and label the face if identified
            if True in result:

                # label face if face is found
                name = face_names[result.index(True)]
                top, right, bottom, left = face_location
                cv2.rectangle(image, (left*scl, top*scl), (right*scl, bottom*scl), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left*scl, bottom*scl + 20), font, 0.8, (255, 255, 255), 1)

                # Find face landmarks using face_recognition library and draw them on the user's face on screen
                face_landmarks_list = face_recognition.face_landmarks(image)

                # loop over each landmark to draw
                for face_landmarks in face_landmarks_list:

                    # loop over each facial feature to draw (ie lips, chin, etc)
                    for facial_feature in face_landmarks.keys():

                        # loop through the points found to draw small lines that draw the facial feature
                        for i in range(0, len(face_landmarks[facial_feature])-2):
                            cv2.line(image, face_landmarks[facial_feature][i], face_landmarks[facial_feature][i+1], (255, 0, 255), 1)


        # quit the program if 'q' key pressed - sometimes a little slow
        cv2.imshow("frame", image)
        k = cv2.waitKey(1)
        if k > 0:
            print(k)
        if k & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break



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


def main():

    print("\nWelcome to Hewitt's Open CV Project!!!")
    file = input("\nPlease input the filename of an image: ")

    # Use tesseract to get words from an image store into text file
    find_Words(file) # give name of image
    print("\nGet your files to find a text file containing all text from the image or look at a preview of it in terminal.") # fix when this prints


    print("\nNow, let's try to figure out who you are.")
    consent = input("Can I figure out who you are? (y/n): ")
    if consent == "y":

        # Use face_recognition library to identify faces using the laptop camera
        print("Press 'q' to quit. Have fun!")
        find_Faces()

main()
