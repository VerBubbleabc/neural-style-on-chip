import cv2
cap=cv2.VideoCapture(0)
sucess,img=cap.read()
cv2.imwrite('im8.jpg',img)