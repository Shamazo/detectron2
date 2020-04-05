import torch
import os
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_resnet_backbone, build_backbone
from detectron2.modeling.meta_arch import build_model
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.engine import SimpleTrainer, HookBase, default_setup, DefaultTrainer
from detectron2.solver import build_optimizer
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from cfpn.cfpn import CFPN

def add_cfpn_config(cfg):
    _C = cfg
    _C.MODEL.RECONSTRUCT_HEADS = CN()
    _C.MODEL.RECONSTRUCT_HEADS.NAME = "SPHead"
    _C.MODEL.RECONSTRUCT_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.RECONSTRUCT_HEADS.IN_CHANNELS = 256

def setup(args):
    cfg = get_cfg()
    add_cfpn_config(cfg)
    cfg.merge_from_file('./configs/subpixel_CFPN_1x.yaml')
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cfpn")
    return cfg

cfg = setup([])
print(cfg)
# cfpn = build_model(cfg)
# opt = build_optimizer(cfg, cfpn)
# voc_dl = build_detection_train_loader(cfg)
trainer = DefaultTrainer(cfg)
# print(trainer.build_hooks())
trainer.train()