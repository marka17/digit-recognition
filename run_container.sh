#!/usr/bin/env bash

image_name=digit-recognition-image:latest
docker pull ${image_name}

username=$(whoami)
container_name=${username}-digit-recognition

nvidia-docker stop ${container_name}
nvidia-docker rm ${container_name}

nvidia-docker run -it -d --net=host --ipc=host \
    -v /home/${username}/digit-recognition:/digit-recognition \
    -v /mnt/DATA/${username}:/data \
    -v /mnt/SSD/${username}:/ssd \
    -e PYTHONPATH=/digit-recognition \
    -w /digit-recognition --name ${container_name} ${image_name} bash