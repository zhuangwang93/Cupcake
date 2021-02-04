model="resnet50"
epochs=5
gpus=$1
fusion=2
script="cifar10_benchmark.py"
log_path="training_speed"

arguments="--compress --compressor fp16 --comm allreduce --memory none --speed-test --epochs ${epochs} --fusion-num ${fusion} --model $model --scheduler"
horovodrun -np ${gpus} python ${script} --speed-test --epochs ${epochs} --model $model | tee ${log_path}/nvlink/${model}/${gpus}GPU_fp32
horovodrun -np ${gpus} python ${script} --fp16 --speed-test --epochs ${epochs} --model $model | tee ${log_path}/nvlink/${model}/${gpus}GPU_fp16
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} ${arguments} | tee ${log_path}/nvlink/${model}/${gpus}GPU_mergeComp
