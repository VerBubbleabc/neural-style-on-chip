import cv2

while True:
    cap = cv2.VideoCapture('output.avi')
    if cap.isOpened():
        break

ret, frame = cap.read()

while ret:
    cv2.imshow('frame', frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) == ord('q'):
        break



