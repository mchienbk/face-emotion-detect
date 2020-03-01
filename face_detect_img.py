# =========================================================================
# Filename:     face_detect_img.py
# Name:         Tran Minh Chien
# Date:         2019.12.22
# =========================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt



def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#  Faces detect function
def face_detect(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)
        
    return image_copy

# Main function
def main():
    # Load Haar cascade data
    haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
    
    # Load image
    img = cv2.imread('test/baby-1.jpg')

    # Detect faces
    faces = face_detect(haar_cascade_face, img)

    # Convert to RGB and display image
    # plt.imshow(convertToRGB(faces))

    # Display the images in subplots
    cv2.imshow('faces',faces)

if __name__ == '__main__':
    main()
    cv2.waitKey(0)