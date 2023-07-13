model="resnet50"
epochs=5
fusion=1
script="cifar10_benchmark.py"
log_path="training_speed"
gpus=$1

compress_arguments="--compress --compressor efsignsgd --comm allgather --memory residual --fusion-num ${fusion}"
horovodrun -np ${gpus} python3 ${script} ${compress_arguments} --epochs ${epochs} --model $model --speed-test 

compress_arguments="--compress --compressor dgc --comm allgather --memory dgc --fusion-num ${fusion} --compress-ratio 0.05"
horovodrun -np ${gpus} python3 ${script} ${compress_arguments} --epochs ${epochs} --model $model --speed-test 