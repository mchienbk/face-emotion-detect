# =========================================================================
# Smile detection in vieo by HAAR-CADCASE and CNN
# Filename:    smile_detect_video.py
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
    model = load_model('models/happy_gray_aug.h5')

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


    while(cap.isOpened()):
        
        #debug 
        i += 1
        print(i)   

        # Get image frame by frame
        ret, img = cap.read()
        input_img = img.copy()
        img2 = img.copy()

        if not ret:
            break

        # img = cv2.resize(img, dsize=(640, int(img.shape[0] * 640 / img.shape[1])))

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
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            padding_size = int(w/1.2)

            x1, y1, x2, y2 = x - padding_size, y - padding_size, x+w + padding_size, y+h + padding_size
            desired_size = max(x2 - x1, y2 - y1)
            # img2 = img.copy()
            # cv2.rectangle(img2, pt1=(x1, y1), pt2=(x2, y2), color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

            # padding
            ones = np.ones_like(input_img)
            ones[y:(y+h), x:(x+w)] = 0
            input_img[np.logical_and(input_img == [[[0,0,0]]], ones.astype(np.bool))] = 140

            x1, y1 = max(0, x1), max(0, y1)
            input_img = input_img[y1:y2, x1:x2]

            delta_w = desired_size - input_img.shape[1]
            delta_h = desired_size - input_img.shape[0]
            top, bottom = delta_h//2, delta_h//2
            left, right = delta_w//2, delta_w//2
           
            # cv2.imwrite("boder_%s.jpg" %i,input_img)
            input_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(140, 140, 140))
            
            # cv2.imwrite("input_%s.jpg" %i,input_img)
            if(len(input_img.shape)>=3):
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("output_%s.jpg" %i,input_img)
            
            input_img = cv2.resize(input_img, dsize=(64, 64)).astype(np.float64)

            input_img_copy = input_img.copy()
            input_img -= np.mean(input_img_copy, keepdims=True)
            input_img /= (np.std(input_img_copy, keepdims=True) + 1e-6)

            pred = model.predict(input_img.reshape(1, 64, 64, 1))
            is_smile = pred[0][0] > 0.5
            
            # cv2.putText(img, 'Smile:', (img.shape[1]//2-100, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)
            cv2.putText(img, 'Smile:', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_4)

            if is_smile:
                # cv2.putText(img, '%s(%s%%)' % (is_smile, int(pred[0][0] * 100)), (img.shape[1]//2, img.shape[0]-50), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 255, 0), 2, cv2.LINE_4)
                cv2.putText(img, '%s(%s%%)' % (is_smile, int(pred[0][0] * 100)), (x+100, y+h), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_4)

            else:
                # cv2.putText(img, '%s(%s%%)' % (is_smile, int(pred[0][0] * 100)), (img.shape[1]//2, img.shape[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(img, '%s(%s%%)' % (is_smile, int(pred[0][0] * 100)), (x+100, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)

        cv2.imshow('img', img)
        cv2.imshow('input_img', input_img)
        video_out.write(img)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # cv2.waitKey(0)