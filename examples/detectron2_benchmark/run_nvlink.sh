gpus=$1
script="mask_rcnn.py"
log_path="training_speed"

config_file="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
arguments="--compress --compressor fp16 --comm allreduce --memory none --fusion-num 2 --scheduler --config-file ${config_file}"

horovodrun -np ${gpus} python ${script} --config-file ${config_file} | tee ${log_path}/nvlink/${gpus}GPU_fp32
horovodrun -np ${gpus} python ${script} --fp16-allreduce --config-file ${config_file} | tee ${log_path}/nvlink/${gpus}GPU_fp16
horovodrun -np ${gpus} --cycle-time-ms 0.1 python ${script} ${arguments} | tee ${log_path}/nvlink/${gpus}GPU_fp16_mergComp
