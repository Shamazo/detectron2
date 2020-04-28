import os
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import SimpleTrainer, HookBase, default_setup, DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, PascalVOCDetectionEvaluator

from cfpn.evaluator import ReconstructionEvaluator
from cfpn.datasets.kodak import download_kodak, register_kodak
from cfpn.meta_arch import RCNNwithReconstruction



def add_theis_config(cfg):
    _C = cfg
    _C.MODEL.THEIS_CAE = CN()
    _C.MODEL.THEIS_CAE.OUT_FEATURE = "cae_encoder_top"
    _C.MODEL.THEIS_CAE.PATCHED  = False
    _C.MODEL.THEIS_CAE.EDGE_LENGTH  = 128
    _C.MODEL.THEIS_CAE.INTERIOR_DIM = 128


    _C.MODEL.RECONSTRUCT_HEADS = CN()
    _C.MODEL.RECONSTRUCT_HEADS_ON = False
    _C.MODEL.RECONSTRUCT_HEADS.NAME = ""
    _C.MODEL.RECONSTRUCT_HEADS.IN_FEATURES = []

    _C.TEST.TEST_IMAGES = ["img_2"]


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg_arg: CfgNode, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        eval_img = cfg.TEST.TEST_IMAGES[0]
        evaluators = []
        # if dataset_name != 'kodak_test':
        #     evaluators = [COCOEvaluator(dataset_name, cfg_arg, True)]
        # else:
        #     evaluators = []
        if cfg.MODEL.RECONSTRUCT_HEADS_ON and dataset_name == 'kodak_test':
            evaluators.append(ReconstructionEvaluator(dataset_name, output_folder, eval_img=eval_img))
        return DatasetEvaluators(evaluators)


def setup(args):
    cfg = get_cfg()
    add_theis_config(cfg)
    cfg.merge_from_file('./configs/Miro-CAE-reconstruct.yaml')
    download_kodak()
    register_kodak()
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cae")
    return cfg


if __name__ == "__main__":
    cfg = setup([])
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
