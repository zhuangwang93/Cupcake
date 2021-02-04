import sys
import torch
import horovod.torch as hvd


def grace_from_params(params):
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allgather')
    model_params = params.get('params', 'none')
    ratio = params.get('ratio', 0.01)
    if model_params == 'none':
        sys.exit("No model parameters for grace_from_params()")
    fusion_num = params.get('fusion_num', 2)
    momentum = params.get('momentum', 0.9)
    qsgd_quantum = params.get('quantum', 64)
    device = torch.cuda.current_device()
    print("[Compression Setup] compressor: {}\n\tmemory: {}\n\tcommunicator: {}\n\tsparsity ratio: {}\n\tfusion num: {}".format(
        comp,
        mem,
        comm,
        ratio,
        fusion_num
    ))


    if comp == 'dgc':
        from mergeComp_dl.torch.compressor.pooldgc import PoolDgcCompressor
        compressor = PoolDgcCompressor(compress_ratio=ratio)
    elif comp == 'efsignsgd':
        from mergeComp_dl.torch.compressor.poolefsignsgd import PoolEFSignSGDCompressor
        compressor = PoolEFSignSGDCompressor(device=device)
    elif comp == 'fp16':
        from mergeComp_dl.torch.compressor.poolfp16 import PoolFP16Compressor
        compressor = PoolFP16Compressor()
    elif comp == 'none':
        from mergeComp_dl.torch.compressor.poolnone import PoolNoneCompressor
        compressor = PoolNoneCompressor()
    elif comp == 'onebit':
        from mergeComp_dl.torch.compressor.poolonebit import PoolOneBitCompressor
        compressor = PoolOneBitCompressor(device=device)
    elif comp == 'qsgd':
        from mergeComp_dl.torch.compressor.poolqsgd import PoolQSGDCompressor
        compressor = PoolQSGDCompressor(quantum_num=qsgd_quantum)
    elif comp == 'randomk':
        from mergeComp_dl.torch.compressor.poolrandomk import PoolRandomKCompressor
        compressor = PoolRandomKCompressor(compress_ratio=ratio)
    elif comp == 'signsgd':
        from mergeComp_dl.torch.compressor.poolsignsgd import PoolSignSGDCompressor
        compressor = PoolSignSGDCompressor(device=device)
    elif comp == 'signum':
        from mergeComp_dl.torch.compressor.poolsignum import PoolSignumCompressor
        compressor = PoolSignumCompressor(momentum=momentum, device=device)
    elif comp == 'terngrad':
        from mergeComp_dl.torch.compressor.poolterngrad import PoolTernGradCompressor
        compressor = PoolTernGradCompressor(device=device)
    elif comp == 'threshold':
        from mergeComp_dl.torch.compressor.poolthreshold import PoolThresholdCompressor
        compressor = PoolThresholdCompressor(threshold=ratio)
    elif comp == 'topk':
        from mergeComp_dl.torch.compressor.pooltopk import PoolTopKCompressor
        compressor = PoolTopKCompressor(compress_ratio=ratio)
    else:
        raise NotImplementedError(comp)

    if fusion_num > 0:
        if mem == 'dgc':
            from mergeComp_dl.torch.memory.pooldgc import PoolDgcMemory
            if comp == 'topk' or comp == 'randomk':
                memory = PoolDgcMemory(model_params, momentum=0, fusion_num=fusion_num, momentum_masking=False)
            else:
                memory = PoolDgcMemory(model_params, fusion_num=fusion_num)
        elif mem == 'none':
            from mergeComp_dl.torch.memory.poolnone import PoolNoneMemory
            memory = PoolNoneMemory(model_params, fusion_num=fusion_num)
        elif mem == 'residual':
            from mergeComp_dl.torch.memory.poolresidual import PoolResidualMemory
            memory = PoolResidualMemory(model_params, fusion_num=fusion_num)
        elif mem == '1bitadam':
            from mergeComp_dl.torch.memory.pool1bitadam import Pool1BitAdamMemory
            memory = Pool1BitAdamMemory(model_params, fusion_num=fusion_num)
        else:
            raise NotImplementedError(mem)
    elif fusion_num == 0:
        if mem == 'dgc':
            from mergeComp_dl.torch.memory.dgc import DgcMemory
            if comp == 'topk' or comp == 'randomk':
                memory = DgcMemory(model_params, momentum=0, momentum_masking=False)
            else:
                memory = DgcMemory(model_params)
        elif mem == 'none':
            from mergeComp_dl.torch.memory.none import NoneMemory
            memory = NoneMemory(model_params)
        elif mem == 'residual':
            from mergeComp_dl.torch.memory.residual import ResidualMemory
            memory = ResidualMemory(model_params)
        else:
            raise NotImplementedError(mem)

    if comm == 'allreduce':
        from mergeComp_dl.torch.communicator.pool_allreduce import PoolAllreduce
        return PoolAllreduce(compressor, memory)
    elif comm == 'allgather':
        from mergeComp_dl.torch.communicator.pool_allgather import PoolAllgather
        return PoolAllgather(compressor, memory)
    elif comm == 'ps':
        from mergeComp_dl.torch.communicator.pool_ps import PoolPS
        return PoolPS(compressor, memory)
    else:
        raise NotImplementedError(comm)



def add_parser_arguments(parser):
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--compress', action='store_true', default=False,
                        help='use gradient compression')
    parser.add_argument('--compressor', type=str, default='efsignsgd',
                        help='compress algorithm')
    parser.add_argument('--compress-ratio', type=float, default=0.01,
                        help='compress ratio for sparsification')
    parser.add_argument('--memory', type=str, default='residual',
                        help='compress algorithm')
    parser.add_argument('--fusion-num', type=int, default=2,
                        help='the number of merged tensors')
    parser.add_argument('--comm', type=str, default='allgather',
                        help='communication for compression')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use Adam optimizer')
    parser.add_argument('--speed-test', action='store_true', default=False,
                        help='test the training speed')

    parser.add_argument('--scheduler', action='store_true', default=False,
                        help='use scheduler partition')
    parser.add_argument('--scheduler-epoch', type=int, default=20,
                        help='scheduler AVERAGE for epoch')
    parser.add_argument('--scheduler-step', type=int, default=2,
                        help='scheduler add step for search')
    parser.add_argument('--scheduler-warmup', type=int, default=5,
                        help='warmup iterations for scheduler')
    parser.add_argument('--scheduler-baseline', action='store_true', default=False,
                        help='use scheduler baseline based on tensor number')
    return parser



def wrap_compress_optimizer(model, optimizer, args):
    if args.compress:
        """
        compressor: dgc, efsignsgd, fp16, none, onebit, qsgd, randomk, signsgd, signum, terngrad, threshold, topk
        memory: dgc, none, residual, 1bitadam.   Note: 1bitadam is for Adam
        comm: allreduce, allgather, ps
        """
        params = {
            'compressor': args.compressor,
            'memory': args.memory,
            'communicator': args.comm,
            'params': model.named_parameters(),
            'ratio': args.compress_ratio,
            'fusion_num': args.fusion_num,
        }

        grc = grace_from_params(params)
        compress_config = (1, 0)

        from .horovod.optimizer import DistributedOptimizer
        optimizer = DistributedOptimizer(optimizer, compression=grc, named_parameters=model.named_parameters(), compress_config=compress_config)

        return optimizer, grc
    else:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)
        return optimizer, None
