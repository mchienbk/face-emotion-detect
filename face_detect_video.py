# =========================================================================
# Filename:     face_detect_img.py
# Name:         Tran Minh Chien
# Date:         2019.12.22
# =========================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

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
    haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

    # Video capture
    # video_capture = cv2.VideoCapture('test/video.mp4')
    video_capture = cv2.VideoCapture(0)

    # # Detect faces
    # faces = face_detect(haar_cascade_face, img)

    # # Display the images in subplots
    # cv2.imshow('faces',faces)

    while True:
    # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = haar_cascade_face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags=cv2.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # cv2.waitKey(0)

