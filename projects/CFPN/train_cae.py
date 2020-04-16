from detectron2.config import CfgNode, get_cfg
from detectron2.engine import SimpleTrainer, HookBase, default_setup, DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, PascalVOCDetectionEvaluator

import cfpn.cae.backbone_hooks


def add_theis_config(cfg):
    _C = cfg
    _C.MODEL.THEIS_CAE = CN()
    _C.MODEL.THEIS_CAE.OUT_FEATURE = "cae_encoder_top"

    _C.MODEL.RECONSTRUCT_HEADS = CN()
    _C.MODEL.RECONSTRUCT_HEADS_ON = False
    _C.MODEL.RECONSTRUCT_HEADS.NAME = ""
    _C.MODEL.RECONSTRUCT_HEADS.IN_FEATURES = []


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg_arg: CfgNode, dataset_name):
        evaluators = [COCOEvaluator(dataset_name, cfg_arg, True)]
        return DatasetEvaluators(evaluators)


def setup(args):
    cfg = get_cfg()
    add_theis_config(cfg)
    cfg.merge_from_file('./configs/Miro-CAE-reconstruct.yaml')
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cae")
    return cfg


if __name__ == "__main__":
    cfg = setup([])
    trainer = Trainer(cfg)
    trainer.train()
