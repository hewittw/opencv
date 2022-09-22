import cv2
from pytesseract import pytesseract

pytesseract.tesseract_cmd = "/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract"

img = cv2.imread("top_artists_short_term.png")

words_in_image = pytesseract.image_to_string(img)

print(words_in_image)
