# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os

from fsdet.data.datasets.coco import load_coco_json

import fsdet.utils.comm as comm
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg, set_global_cfg
from fsdet.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog

from fsdet.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from fsdet.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    verify_results,
)

# Dataset Root
DATASET_ROOT = "/home/eric/mmdetection/data/VOCdevkit/datasets/set1/comparison/" #need change
DATASET_ROOT_few = "/home/eric/mmdetection/data/VOCdevkit/datasets/set1/balanced60/" #need change
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
ANN_ROOT_few = os.path.join(DATASET_ROOT_few, 'annotations')

TRAINVALTEST_PATH = os.path.join(DATASET_ROOT_few, 'trainvaltest')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')

TRAINVALTEST_JSON = os.path.join(ANN_ROOT_few, 'instances_trainvaltest.json')    
TEST_JSON = os.path.join(ANN_ROOT, 'instances_test.json') 

# take images out from the whole dataset
from json_handler import json_handler
if(os.path.isdir(TRAINVALTEST_PATH)==False):
    js = json_handler(
    jpg_data_root= "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/",
    coco_data_root = DATASET_ROOT_few, subset = 'trainvaltest')
    js.write_jpg_txt()
    js.get_jpg_from_txt()

def plain_register_dataset():
    DatasetCatalog.register("train_tea", lambda: load_coco_json(TRAINVALTEST_JSON, TRAINVALTEST_PATH, "train_tea"))
    MetadataCatalog.get("train_tea").set(thing_classes=['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 'miner', 'thrips', 'tetrany', 'formosa', 'other'],
                                                    json_file=TRAINVALTEST_JSON,
                                                    image_root=TRAINVALTEST_PATH)

    DatasetCatalog.register("test_tea", lambda: load_coco_json(TEST_JSON, TEST_PATH, "test_tea"))
    MetadataCatalog.get("test_tea").set(thing_classes= ['brownblight', 'algal', 'blister', 'sunburn', 'fungi_early', 'roller', 'moth', 'tortrix', 'flushworm', 'caloptilia', 'mosquito_early', 'mosquito_late', 'miner', 'thrips', 'tetrany', 'formosa', 'other'],
                                                json_file=TEST_JSON,
                                                image_root=TEST_PATH)

# from fsdet.data.dataset_mapper import AlbumentationMapper

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = None
        # if cfg.INPUT.USE_ALBUMENTATIONS:
        #     mapper = AlbumentationMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("train_tea",)
    cfg.DATASETS.TEST = ("test_tea",)
    cfg.OUTPUT_DIR = "/home/eric/FSCE_tea-diseases/checkpoints/coco/faster_rcnn/set1/split0/60shot/" #need change
    cfg.SOLVER.IMS_PER_BATCH = 3  # batch_size; 
    # ITERS_IN_ONE_EPOCH = int(480 / cfg.SOLVER.IMS_PER_BATCH) #need change; iters_in_one_epoch = dataset_imgs/batch_size 
    # cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 24) - 1 # epochs
    # cfg.SOLVER.STEPS = (ITERS_IN_ONE_EPOCH*10, ITERS_IN_ONE_EPOCH*16, ITERS_IN_ONE_EPOCH*20)
    # cfg.SOLVER.GAMMA = 0.2
    # cfg.TEST.EVAL_PERIOD = 4*ITERS_IN_ONE_EPOCH 

    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)
    plain_register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    import shutil
    shutil.rmtree(TRAINVALTEST_PATH)
