model="resnet50"
epochs=5
script="cifar10_benchmark.py"
log_path="training_speed"
gpus=$1

horovodrun -np ${gpus} python ${script} --speed-test --epochs ${epochs} --model $model | tee ${log_path}/${model}/${gpus}GPU_baseline
