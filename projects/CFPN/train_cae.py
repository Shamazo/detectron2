import torch
import os
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_resnet_backbone, build_backbone
from detectron2.modeling.meta_arch import build_model
from detectron2.config import CfgNode, get_cfg
from detectron2.layers import ShapeSpec
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.engine import SimpleTrainer, HookBase, default_setup, DefaultTrainer
from detectron2.solver import build_optimizer
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, PascalVOCDetectionEvaluator
from cfpn.cfpn import CFPN
from cfpn.datasets.kodak import download_kodak, register_kodak

import cfpn.cae.backbone_hooks

def add_theis_config(cfg):
    _C = cfg
    _C.MODEL.THEIS_CAE = CN()
    _C.MODEL.THEIS_CAE.OUT_FEATURE = "cae_encoder_top"

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name):
        evaluators = [COCOEvaluator(dataset_name, cfg, True)]
        return DatasetEvaluators(evaluators)


def setup(args):
    cfg = get_cfg()
    add_theis_config(cfg)
    cfg.merge_from_file('./configs/Miro-Cae.yaml')
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cae")
    return cfg

if __name__ == "__main__":
    cfg = setup([])
    trainer = Trainer(cfg)
    trainer.train()