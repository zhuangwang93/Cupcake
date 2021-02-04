script="mask_rcnn.py"
log_path="training_speed"
gpus=$1

arguments="--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
horovodrun -np ${gpus} python ${script} ${arguments} | tee ${log_path}/${gpus}GPU_baseline
