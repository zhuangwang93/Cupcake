model="resnet101"
epochs=5
gpus=$1
fusion=$2
mergeComp=$3
script="pytorch_imagenet.py"
log_path="training_speed"
sch=""

if [[ $fusion -eq 1 ]]
then
    wise="model_wise"
elif [[ $fusion -eq 0 ]]
then
    wise="layer_wise"
else
    if [[ $mergeComp -eq 1 ]]
    then
        wise="mergeComp"
    elif [[ $mergeComp -eq 2 ]]
    then
        wise="number_baseline"
    else
        wise="size_baseline"
    fi
fi


compress_arguments="--compress --compressor efsignsgd --comm allgather --memory residual --fusion-num ${fusion}"
horovodrun -np ${gpus} python3 ${script} ${compress_arguments} --epochs ${epochs} --model $model --speed-test 

compress_arguments="--compress --compressor dgc --comm allgather --memory dgc --fusion-num ${fusion} --compress-ratio 0.01"
horovodrun -np ${gpus} python3 ${script} ${compress_arguments} --epochs ${epochs} --model $model --speed-test