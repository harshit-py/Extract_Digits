import numpy as np
import cv2


def preprocess(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 200 < cv2.contourArea(cnt) < 1000 and 0.2 < w/h < 5:
            digit_contours.append(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img, digit_contours


if __name__ == '__main__':
    img_path = 'data/Screenshot 2024-04-19 at 7.35.49 PM.png'
    img, _ = preprocess(img_path)
    cv2.imshow('img', img)
    cv2.waitKey(0)