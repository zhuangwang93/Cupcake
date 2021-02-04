 horovodrun -np 2 --cycle-time-ms 0.1 python train.py -net resnet152 -b 128 -lr 0.1 --compress --compressor efsignsgd --fusion-num 1 | tee logs2/efsignsgd_mw_resnet152
 horovodrun -np 2 --cycle-time-ms 0.1 python train.py -net resnet152 -b 128 -lr 0.1 --compress --compressor efsignsgd --fusion-num 3 | tee logs2/efsignsgd_3_resnet152
 horovodrun -np 2 python train.py -net resnet152 -b 128 -lr 0.1 --compress --compressor efsignsgd --fusion-num 0 | tee logs2/efsignsgd_lw_resnet152
