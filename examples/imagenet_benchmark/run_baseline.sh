model="resnet101"
epochs=5
script="pytorch_imagenet.py"
log_path="training_speed"
gpus=$1

horovodrun -np ${gpus} python ${script} --speed-test --epochs ${epochs} --model $model | tee ${log_path}/${model}/${gpus}GPU_baseline
