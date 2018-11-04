#!/bin/sh
for i in $(seq 3);do
    sudo fswebcam -d /dev/video0 /home/xilinx/jupyter_notebooks/style/im3.jpg
    sleep 1
done;