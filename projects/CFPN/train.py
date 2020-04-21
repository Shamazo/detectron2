import logging
import os
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import default_setup, DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, DatasetEvaluator, inference_on_dataset, \
    print_csv_format
from collections import OrderedDict
from cfpn.evaluator import ReconstructionEvaluator
from cfpn.datasets.kodak import download_kodak, register_kodak


def add_cfpn_config(cfg):
    _C = cfg
    _C.MODEL.RECONSTRUCT_HEADS = CN()
    _C.MODEL.RECONSTRUCT_HEADS_ON = False
    _C.MODEL.RECONSTRUCT_HEADS.NAME = "SPHead"
    _C.MODEL.RECONSTRUCT_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.RECONSTRUCT_HEADS.IN_CHANNELS = 256
    #how much to weight the reconstruction at each level by
    _C.MODEL.RECONSTRUCT_HEADS.LOSS_WEIGHTS = [1, 1, 1, 1, 1]
    _C.TEST.TEST_IMAGES = ["img_2"]
    _C.MODEL.QUANTIZER_ON = True
    _C.MODEL.QUANTIZER = CN()
    _C.MODEL.QUANTIZER.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    # These are the weights for the loss functions
    # Must be the same length as in_features and correspond one to one
    _C.MODEL.QUANTIZER.FEAT_WEIGHTS = [10, 10, 10, 10]
    _C.MODEL.QUANTIZER.NAME = 'GSM'

    _C.MODEL.QUANTIZER_ON = False
    _C.MODEL.QUANTIZER = CN()
    _C.MODEL.QUANTIZER.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    # These are the weights for the loss functions
    # Must be the same length as in_features and correspond one to one
    _C.MODEL.QUANTIZER.FEAT_WEIGHTS = [1, 1, 1, 1]
    _C.MODEL.QUANTIZER.NAME = 'GSM'

    _C.TEST.NUM_COMPRESSION_IMAGES = 1000


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name, model=None):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        eval_img = cfg.TEST.TEST_IMAGES[0]
        if dataset_name != 'kodak_test':
            evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        else:
            evaluators = []
        if cfg.MODEL.RECONSTRUCT_HEADS_ON and dataset_name == 'kodak_test':
            evaluators.append(ReconstructionEvaluator(dataset_name, output_folder, eval_img=eval_img))
        return DatasetEvaluators(evaluators)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, model=model)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results




def setup(args):
    cfg = get_cfg()
    add_cfpn_config(cfg)
    cfg.merge_from_file('/home/hamish/detectron2/projects/CFPN/configs/quantized_multilevel_residual_CFPN_1x.yaml')
    # cfg.merge_from_list(args.opts)
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
