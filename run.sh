horovodrun -np 8 -hostfile hostfile python3 compress_benchmark.py --comm allreduce --model=resnet50
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor efsignsgd --comm allgather --memory residual --fusion-num 2 --model=resnet50
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor signsgd --comm allgather --memory residual --fusion-num 2 --model=resnet50
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor onebit --comm allgather --memory residual --fusion-num 2 --model=resnet50
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor signum --comm allgather --memory residual --fusion-num 2 --model=resnet50
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor terngrad --comm allgather --memory residual --fusion-num 2 --model=resnet50
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor qsgd --comm allgather --memory residual --fusion-num 2 --model=resnet50

# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor randomk --comm allgather --memory residual --fusion-num 2 --model=resnet50 --compress-ratio 0.01
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor topk --comm allgather --memory residual --fusion-num 2 --model=resnet50 --compress-ratio 0.01
# horovodrun -np 8 python3 compress_benchmark.py --compress --compressor dgc --comm allgather --memory dgc --fusion-num 2 --model=resnet50 --compress-ratio 0.01