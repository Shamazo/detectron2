import os
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import default_setup, DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

from cfpn.evaluator import CompressionEvaluator
from cfpn.datasets.kodak import download_kodak, register_kodak


def add_cfpn_config(cfg):
    _C = cfg
    _C.MODEL.RECONSTRUCT_HEADS = CN()
    _C.MODEL.RECONSTRUCT_HEADS_ON = False
    _C.MODEL.RECONSTRUCT_HEADS.NAME = "SPHead"
    _C.MODEL.RECONSTRUCT_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.RECONSTRUCT_HEADS.IN_CHANNELS = 256
    _C.MODEL.RECONSTRUCT_HEADS.OUTPUT_IMAGES = ['img_2', 'img_3', 'img_4', 'img_5', 'img_6']
    #how much to weight the reconstruction at each level by
    _C.MODEL.RECONSTRUCT_HEADS.LOSS_WEIGHTS = [1, 1, 1, 1, 1]
    _C.TEST.TEST_IMAGES = ["img_2"]

    _C.MODEL.QUANTIZER_ON = False
    _C.MODEL.QUANTIZER = CN()
    _C.MODEL.QUANTIZER.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    # These are the weights for the loss functions
    # Must be the same length as in_features and correspond one to one
    _C.MODEL.QUANTIZER.FEAT_WEIGHTS = [1, 1, 1, 1]
    _C.MODEL.QUANTIZER.NAME = 'GSM'


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        eval_img = cfg.TEST.TEST_IMAGES[0]
        if dataset_name != 'kodak_test':
            evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        else:
            evaluators = []
        if cfg.MODEL.RECONSTRUCT_HEADS_ON and dataset_name == 'kodak_test':
            evaluators.append(CompressionEvaluator(dataset_name, output_folder, eval_img=eval_img))
        return DatasetEvaluators(evaluators)


def setup(args):
    cfg = get_cfg()
    add_cfpn_config(cfg)
    cfg.merge_from_file('/home/hamish/cs265/detectron2/projects/CFPN/configs/quantized_multilevel_residual_CFPN_1x.yaml')
    download_kodak()
    register_kodak()
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cfpn")
    return cfg


if __name__ == "__main__":
    cfg = setup([])
    trainer = Trainer(cfg)
    trainer.train()
