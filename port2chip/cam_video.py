import cv2
cap = cv2.VideoCapture(0)
ret,img = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = cv2.VideoWriter('video.avi', fourcc, fps, size)

num_frame = 1

while ret:
    out.write(img)
    print('Frame {}'.format(num_frame))
    num_frame += 1
    ret, img = cap.read()
    if num_frame >= 200: break
    if cv2.waitKey(1) == ord('q'):
        print('end')
        break

cap.release()
out.release()
cv2.destroyAllWindows()


