# Usage: bash auto_run_pcie {gpus} {fusion-num} {mergeComp}

gpus=$1
fusion=$2
mergeComp=$3
script="mask_rcnn.py"
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
        sch="--scheduler --scheduler-epoch 20"
    elif [[ $mergeComp -eq 2 ]]
    then
        wise="number_baseline"
        sch="--scheduler --scheduler-baseline"
    else
        wise="size_baseline"
    fi
fi

arguments="--compress --fusion-num ${fusion} ${sch} --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor fp16 --memory none --comm allreduce | tee ${log_path}/${wise}/${gpus}GPU/fp16
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor efsignsgd | tee ${log_path}/${wise}/${gpus}GPU/efsignsgd
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor qsgd --memory none | tee ${log_path}/${wise}/${gpus}GPU/qsgd
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor signsgd --memory none | tee ${log_path}/${wise}/${gpus}GPU/signsgd
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor signum --memory none | tee ${log_path}/${wise}/${gpus}GPU/signum
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor onebit --memory none | tee ${log_path}/${wise}/${gpus}GPU/onebit
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor dgc --memory dgc | tee ${log_path}/${wise}/${gpus}GPU/dgc
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor topk --memory dgc | tee ${log_path}/${wise}/${gpus}GPU/topk
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} $arguments --compressor randomk --memory dgc | tee ${log_path}/${wise}/${gpus}GPU/randomk
