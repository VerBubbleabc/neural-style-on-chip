# use 4layer version for better visual effect
import cv2
import numpy as np


z = np.load('style_full.npz')
w_conv1 = z['conv1.weight']
b_conv1 = z['conv1.bias']
w_conv2 = z['conv2.weight']
b_conv2 = z['conv2.bias']
w_conv3 = z['conv3.weight']
b_conv3 = z['conv3.bias']
w_conv4 = z['conv4.weight']
b_conv4 = z['conv4.bias']
w_conv5 = z['conv5.weight']
b_conv5 = z['conv5.bias']
w_conv6 = z['conv6.weight']
b_conv6 = z['conv6.bias']
w_conv7 = z['conv7.weight']
b_conv7 = z['conv7.bias']
w_conv8 = z['conv8.weight']
b_conv8 = z['conv8.bias']
conv_param_x3 = {'stride': 1, 'pad': 1}
conv_param_x1 = {'stride': 1, 'pad': 0}

from cs231n.fast_layers import conv_forward_im2col
conv_forward_fast = conv_forward_im2col

def forward(im):
    x = conv_forward_fast(im, w_conv1, b_conv1, conv_param=conv_param_x3)
    x = np.maximum(x, 0)
    x = conv_forward_fast(x, w_conv2, b_conv2, conv_param=conv_param_x1)
    x = np.maximum(x, 0)
    x = conv_forward_fast(x, w_conv3, b_conv3, conv_param=conv_param_x1)
    x = np.maximum(x, 0)
    x = conv_forward_fast(x, w_conv4, b_conv4, conv_param=conv_param_x3)
    x = np.maximum(x, 0) + im
    x = conv_forward_fast(x, w_conv5, b_conv5, conv_param=conv_param_x3)
    x = np.maximum(x, 0)
    x = conv_forward_fast(x, w_conv6, b_conv6, conv_param=conv_param_x1)
    x = np.maximum(x, 0)
    x = conv_forward_fast(x, w_conv7, b_conv7, conv_param=conv_param_x1)
    x = np.maximum(x, 0)
    x = conv_forward_fast(x, w_conv8, b_conv8, conv_param=conv_param_x3)
    oui = np.maximum(x + im, 0)
    return oui

while True:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        break

# main loop here
while True:
    # read in the frame
    ret, frame = cap.read()
    # preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    frame = np.array(frame, dtype=np.float32)
    frame = frame[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    
    oui = forward(frame)
    ou = oui.reshape((3, 128, 128))
    ou = ou.transpose(1, 2, 0)
    ou = np.array(ou, dtype=np.uint8)
    ou = cv2.resize(ou, (256, 256))
    cv2.imshow('frame', ou)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
        
        