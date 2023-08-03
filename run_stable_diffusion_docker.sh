#/bin/bash
export MY_CONTAINER="stable_diffusion"
IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.15.0-torch1.9-ubuntu18.04-py37

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ]; then
    docker run  -e DISPLAY=unix$DISPLAY --net=host --pid=host --ipc=host \
        -v /sys/kernel/debug:/sys/kernel/debug \
        -it --privileged --shm-size 64g \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        --device /dev/cambricon_ctl \
        --name $MY_CONTAINER \
        -v /home:/home \
        -v /datastes:/datastes \
        $IMAGE_NAME /bin/bash
else
    docker start $MY_CONTAINER
    docker exec -ti $MY_CONTAINER /bin/bash
fi



