# =========================================================================
# Emotions detection in vieo by HAAR-CADCASE and CNN
# Filename:    emotion_detect_video.py
# Name:         Tran Minh Chien
# Date:         2019.12.22
# =========================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from keras.models import load_model


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
    model = load_model('models/model_2.h5')

    # Get Video capture
    # cap = cv2.VideoCapture(0)
    video_path = 'test/video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Making save
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_out = cv2.VideoWriter('video_output.mp4', fourcc, cap.get(
        cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # debug
    i = 0

    _feel = ('angry','disgust','fear','happy','neutral','sad','surprise')

    while(cap.isOpened()):
        
        #debug 
        i += 1
        print('frame',i)   

        # Get image frame by frame
        ret, img = cap.read()
        input_img = img.copy()
        show_img = img.copy()
        crop_img = img.copy()
        
        if not ret:
            break

        # Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade_face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags=cv2.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(show_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Calculate padding and resize in input of CNN
            # padding_size = int(w/1.2)
            # x1, y1, x2, y2 = x - padding_size, y - padding_size, x+w + padding_size, y+h + padding_size
            # x1, y1 = max(0, x1), max(0, y1)  
            # x2, y2 = min(x2,img.shape[1]), min(y2,img.shape[0])    
            # desired_size = max(x2 - x1, y2 - y1)

            # if desired_size > min(img.shape[0],img.shape[1]):
            #     desired_size = min(img.shape[0],img.shape[1])

            # delta_w = img.shape[1] - desired_size
            # delta_h = img.shape[0] - desired_size

            # y_new, x_new = int(delta_h/2), int(delta_w/2)

            # crop_img = img[y_new:y_new+desired_size, x_new:x_new+desired_size]



            # Input Image of CNN
            crop_img = img[y:y+h, x:x+w]
            input_img = cv2.resize(crop_img, dsize=(48, 48)).astype(np.float64)

            # Prediction
            pred = model.predict(input_img.reshape(1, 48, 48, 3))
            print(pred)     #debug
            _index = np.argmax(pred) 

            cv2.putText(show_img, 'Feeling:', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
            cv2.putText(show_img, _feel[_index], (x+150, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)


        cv2.imshow('Video input', show_img)
        cv2.imshow('crop_img', crop_img)

        video_out.write(show_img)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # cv2.waitKey(0)