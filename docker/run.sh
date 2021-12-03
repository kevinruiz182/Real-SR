#!/bin/bash

WS_DIR_PATH=$(realpath "$PWD/../codes")

docker run -ti \
--gpus=all \
--privileged=true \
--cap-add=CAP_SYS_ADMIN \
--ipc=host \
-v /home/kvargas/kian/datasets:/root/.datasets \
-v $WS_DIR_PATH:/workspace/rsr \
-p 8080:8080 \
--name rsr-run rsr:latest \
/bin/bash


