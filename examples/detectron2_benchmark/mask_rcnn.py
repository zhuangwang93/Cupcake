#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch

#import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

# compression
import torch.optim as optim
import horovod.torch as hvd
import torch.backends.cudnn as cudnn
import sys
from time import time as time_

logger = logging.getLogger("detectron2")
logger.disabled = True
logger.propagate = False


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def memory_partition():
    with EventStorage(10) as storage:
        optimizer.zero_grad()
        loss_dict = model(partition_inputs)
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        torch.cuda.synchronize()
        start_time = time_()

        losses.backward()
        optimizer.step()

    torch.cuda.synchronize()
    return time_() - start_time


def do_train(cfg, args, model, resume=False):
    # default batch size is 16
    model.train()

    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    max_iter = cfg.SOLVER.MAX_ITER
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        #if comm.is_main_process()
        #else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    #logger.info("Starting training from iteration {}".format(start_iter))

    iters = 0
    iter_cnt = 0
    iter_sample_start = 1
    iter_sample_end = 20
    iter_end = 300
    start_time, end_time = 0, 0
    sample_iters = iter_sample_end - iter_sample_start + 1


    if args.scheduler:
        if args.scheduler_baseline:
            grc.memory.clean()
            grc.compressor.clean()
            grc.memory.partition()
        else:
            from mergeComp_dl.torch.scheduler.scheduler import Scheduler
            Scheduler(grc, memory_partition, args)


    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iters += 1
            iter_cnt += 1
            if iters == iter_end:
                break

            if hvd.local_rank() == 0 and iter_cnt == iter_sample_start:
                torch.cuda.synchronize()
                start_time = time_()

            storage.iter = iteration
            #torch.cuda.synchronize()
            #iter_start_time = time_()

            loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            #torch.cuda.synchronize()
            #iter_model_time = time_()

            #loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            #if comm.is_main_process():
            #    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            #print("loss dict:", loss_dict, "losses:", losses, "reduced loss dict:", loss_dict_reduced, "reduced losses:", losses_reduced)
            losses.backward()

            #torch.cuda.synchronize()
            #iter_backward_time = time_()

            optimizer.step()
            optimizer.zero_grad()

            #torch.cuda.synchronize()
            #print("Iteration: {}\tmodel time: {:.3f} \tbackward time: {:.3f}\tFP+BP Time: {:.3f}\tstep time: {:.3f}\tData size: {}".format(
            #    iteration,
            #    (iter_model_time - iter_start_time),
            #    (iter_backward_time - iter_model_time),
            #    (iter_backward_time - iter_start_time),
            #    time_() - iter_start_time,
            #    len(data)))

            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if args.compress:
                grc.memory.update_lr(optimizer.param_groups[0]['lr'])

            if hvd.local_rank() == 0 and iter_cnt == iter_sample_end:
                torch.cuda.synchronize()
                end_time = time_()
                iter_cnt = 0
                print("Iterations: {}\tTime: {:.3f} s\tTraining speed: {:.3f} iters/s".format(
                    sample_iters,
                    end_time - start_time,
                    sample_iters/(end_time - start_time)
                ))

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                #comm.synchronize()

            #if iteration - start_iter > 5 and (
            #    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            #):
            #    for writer in writers:
            #        writer.write()
            #periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(cfg, args):
    #logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    do_train(cfg, args, model, resume=args.resume)
    #return do_test(cfg, model)



if __name__ == "__main__":
    hvd.init()
    parser = default_argument_parser()
    parser.add_argument('--batch-size', type=int, default=2,
                        help='batch size for schduler')

    sys.path.append("../../")
    from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer
    parser = add_parser_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    #device = torch.device("cuda")
    torch.set_num_threads(2)

    print("Command Line Args:", args)
    cfg = setup(args)
    model = build_model(cfg)

    data_loader = build_detection_train_loader(cfg)
    # data for memory partition
    #partition_inputs, partition_targets = None, None
    for inputs in data_loader:
        partition_inputs = inputs
        break

    optimizer = build_optimizer(cfg, model)
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer, grc = wrap_compress_optimizer(model, optimizer, args)

    main(cfg, args)
