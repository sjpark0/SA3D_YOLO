xhost +local:docker
docker run --shm-size=64g --gpus all -it --rm -p 8050:8050 --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix --volume /:/host --workdir /host$PWD sa3d_yolo
