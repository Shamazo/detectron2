from detectron2.evaluation import DatasetEvaluator
import torch
import torch.nn.functional as F
import detectron2.data.transforms as T
from detectron2.data import build_detection_train_loader
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from collections import defaultdict
from typing import List
import numpy as np
import range_coder as rc
import os


class ReconstructionEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, eval_img="img_2"):
        """
        Args:
            dataset_name: must be 'kodak_test' for now
            output_dir:
            eval_img: the key in the output which contains the image we are evaluating
                Currently hard coding to img_2 which is the largest, but in the future
                we could also compare the lower resolution reproductions
        """
        assert dataset_name == 'kodak_test', "Can only evaluate compression on kodak_test"
        self.dataset_name = dataset_name
        self._output_dir = output_dir
        self.eval_img = eval_img
        self.transform = T.ResizeShortestEdge(
            [512, 512], 512
        )

    def reset(self):
        self.ssim_vals = []
        self.ms_ssim_vals = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a CFPN model. input dicts must contain an 'image' key

            outputs: the outputs of a CFPN model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
                The :class:`Instances` object needs to have `densepose` field.
        """
        assert (len(inputs) == 1)
        # reshape the input image to have a maximum length of 512 as the model preprocesses
        # Much shuffling of data, but also the dataset is only 24 images
        orig_image = inputs[0]['image'].permute(1, 2, 0)
        orig_image = self.transform.get_transform(orig_image).apply_image(orig_image.numpy())
        orig_image = torch.tensor(orig_image).permute(2, 0, 1)
        orig_image = torch.unsqueeze(orig_image, dim=0).float().to(outputs[self.eval_img].get_device())

        reconstruct_image = outputs[self.eval_img].float()
        reconstruct_image = reconstruct_image[:, :, 0:orig_image.shape[2], 0:orig_image.shape[3]]
        assert orig_image.shape[1] == 3, "original image must have 3 channels"
        assert reconstruct_image.shape[1] == 3, "reconstructed image must have 3 channels"
        with torch.no_grad():
            ssim_val = ssim(reconstruct_image, orig_image, data_range=255, size_average=False)
            ms_ssim_val = ms_ssim(reconstruct_image, orig_image, data_range=255, size_average=False)
            self.ssim_vals.extend(ssim_val)
            self.ms_ssim_vals.extend(ms_ssim_val)

    def evaluate(self):
        mean_ssim = torch.mean(torch.stack(self.ssim_vals)).cpu().numpy()
        mean_ms_ssim = torch.mean(torch.stack(self.ms_ssim_vals)).cpu().numpy()
        return {'image-sim': {"ssim": mean_ssim, "ms-ssim": mean_ms_ssim}}


class CompressionEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name, model=None):
        """
        Args:
            model: model to be evaluated.
        """
        assert dataset_name == 'kodak_test', "Can only evaluate compression \
                on kodak_test not on {}".format(dataset_name)
        self.train_loader = build_detection_train_loader(cfg)
        self.test_codes = defaultdict(list)
        self.model = model.backbone.eval()
        self.code_feats = cfg.MODEL.QUANTIZER.IN_FEATURES
        self.num_train_images = cfg.TEST.NUM_COMPRESSION_IMAGES
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.output_dir = cfg.OUTPUT_DIR
        return

    def reset(self):
        self.cdf = None
        self.test_codes = dict()
        self.test_pixels = dict()
        return

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a compressive model
            outputs: the outputs of a compressive model given those inputs, must
                contain the keys listed in cfg.MODEL.QUANTIZER.IN_FEATURES
        """
        for output, idx in enumerate(outputs):
            self.test_codes["img_{}".format(idx)] = []
            num_pixels = inputs[0]['image'].shape[1] * inputs[0]['image'].shape[0]
            self.test_pixels["img_{}".format(idx)] = num_pixels
            for code_feat in self.code_feats:
                self.test_codes["img_{}".format(idx)].append(outputs[code_feat])
        return

    def evaluate(self):
        """
        First we build the latent distribution on a subset of the training set
        then we compress our testset, writing to file and then we see how many bits
        the file is on disk, returning an average.
        Returns:

        """
        self.build_latent_distribution()
        bpp = []
        for key, codes in self.test_codes.items():
            savepath = os.path.join(self.output_dir, key + ".dci")
            self.compress_image(codes, savepath)
            bpp.append(os.stat(savepath).st_size * 8 / self.test_pixels[key])

        return {"bpp": np.mean(bpp)}

    def build_latent_distribution(self, alpha: int = 1):
        """Two passes:num_images_pixels
            1. we calculate the minimum latent value across our entire training
                distribution,
            2. we then add |min| to the latent values such that they are all >= 0 and
                then use torch.bincount to get discrete value counts
          -> which we then laplace smooth and convert into a CDF.
        """
        self.cdf = dict()
        self.min_val = dict()
        for code_feat in self.code_feats:
            self.min_val[code_feat] = torch.tensor(0.0).to(self.device).long()
        num_images = 0
        self.model.eval()
        for batch in self.train_loader:
            with torch.no_grad():
                out_dict = self.model(batch)
                for code_feat in self.code_feats:
                    self.min_val[code_feat] = torch.min(
                        self.min_val[code_feat],
                        out_dict[code_feat].min(),
                    )
            num_images += len(batch)
            if num_images > self.num_train_images:
                break

        for key in self.min_val:
            self.min_val[key] = self.min_val[key].abs()

        self.bins = dict()
        for code_feat in self.code_feats:
            self.bins[code_feat] = torch.tensor(0.0).to(self.device).long()
        num_images = 0
        for batch in self.train_loader:
            with torch.no_grad():
                out_dict = self.model(batch)

                for code_feat in self.code_feats:
                    out_dict[code_feat] = out_dict[code_feat].long() + self.min_val[code_feat]

                    batch_bins = torch.bincount(out_dict[code_feat].flatten())
                    if len(batch_bins) > len(self.bins[code_feat]):
                        batch_bins[: len(self.bins[code_feat])] += self.bins[code_feat]
                        self.bins[code_feat] = batch_bins
                    elif len(self.bins[code_feat]) > len(batch_bins):
                        self.bins[code_feat][: len(batch_bins)] += batch_bins
                    else:
                        self.bins[code_feat] += batch_bins
            num_images += len(batch)
            if num_images > self.num_train_images:
                break

        for code_feat in self.code_feats:
            bins = self.bins[code_feat].float()
            bins_smooth = (
                        (bins + alpha) / (bins.sum() + len(bins) * alpha)).cpu()  # additive smooth counts using alpha
            self.cdf[code_feat] = rc.prob_to_cum_freq(bins_smooth, resolution=2 * len(bins_smooth))  # convert pdf -> cdf

    def compress_image(self, code_list: List[torch.Tensor], savepath="default.dci"):
        """
        Args:
            code_list: List of encodings for image
            savepath: path to save the image to
        Returns: Nothing, writes compressed image savepath
        """

        if not self.cdf:
            raise Exception(
                "Cannot compress image without cdf function (hint: call build_latent_distribution)"
            )
        for x in code_list:
            # Holds each flattened featured to be concatenated after the loop
            code_list = []
            for feat in self.code_feats:
                assert x[feat] == 1, "can only compress one image at a time, for now"
                code_list.append(x[feat].long().flatten())
            full_code = torch.cat(code_list)
            # use RangeEncoder to write compressed representation to file
            encoder = rc.RangeEncoder(savepath)
            encoder.encode(full_code.tolist(), self.cdf)
            encoder.close()

