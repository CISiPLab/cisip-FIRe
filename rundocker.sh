mapdir="$(pwd):/workspace"
mapdata="/data/datasets:/data/datasets"
mapcache="/home/app/.cache:/.cache"
GPU=\"device=$1\"
echo $GPU
docker run -t --rm --gpus ${GPU} -v ${mapdir} -v ${mapdata} -v ${mapcache} --ipc=host --user $(id -u):$(id -g) pt1.4 ${@:2}
