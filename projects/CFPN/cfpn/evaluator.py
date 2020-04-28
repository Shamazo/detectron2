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
import copy


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
        # TODO use same images for min value and cdf calc
        val_cfg = copy.deepcopy(cfg)
        coco_2014_minival
        self.train_loader = iter(build_detection_train_loader(cfg))
        self.test_codes = defaultdict(list)
        self.model = model.eval()
        self.code_feats = cfg.MODEL.QUANTIZER.IN_FEATURES
        self.num_train_images = cfg.TEST.NUM_COMPRESSION_IMAGES
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.output_dir = cfg.OUTPUT_DIR
        self.negative_codes = cfg.TEST.NEGATIVE_CODES
        return

    def reset(self):
        self.cdf = None
        # dict img_key -> list of features
        self.test_codes = dict()
        # number of pixels in an image, does not include padding.
        self.test_pixels = dict()
        # number of seen images in this evaluation, used for keys/filenames
        self.seen_test_images = 0
        return

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a compressive model
            outputs: the outputs of a compressive model given those inputs, must
                contain the keys listed in cfg.MODEL.QUANTIZER.IN_FEATURES
        """
        key = "img_{}".format(self.seen_test_images)
        self.test_pixels[key] = inputs[0]['image'].shape[1] * inputs[0]['image'].shape[2]
        self.test_codes[key] = []
        for code_feat in self.code_feats:
            self.test_codes[key].append(outputs[code_feat])
        self.seen_test_images += 1
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
        print("in evaluate built distribution")
        for key, codes in self.test_codes.items():
            print(key)
            savepath = os.path.join(self.output_dir, key + ".dci")
            self.compress_image(codes, savepath)
            print("File size bits ", os.stat(savepath).st_size * 8 )
            print("Num pixels ", self.test_pixels[key])
            bpp.append(os.stat(savepath).st_size * 8 / self.test_pixels[key])

        return {'Compression': {"bpp": np.mean(bpp)}}

    def build_latent_distribution(self, alpha: int = 1):
        """Two passes:num_images_pixels
            1. we calculate the minimum latent value across our entire training
                distribution,
            2. we then add |min| to the latent values such that they are all >= 0 and
                then use torch.bincount to get discrete value counts
          -> which we then laplace smooth and convert into a CDF.
          If a code is in multiple parts, e.g lateral FPN features, they are flattened and concatenated.
        """
        self.cdf = dict()
        self.min_val = torch.tensor(0.0).to(self.device).long()
        num_images = 0
        self.model.eval()
        if self.negative_codes:
            for batch in self.train_loader:
                with torch.no_grad():
                    out_dict = self.model(batch)
                    for code_feat in self.code_feats:
                        self.min_val = torch.min(
                            self.min_val,
                            out_dict[code_feat].long().min(),
                        )
                num_images += len(batch)
                if num_images > self.num_train_images:
                    break

        self.min_val = self.min_val.abs()
        self.bins = torch.tensor([0.0]).to(self.device).long()
        num_images = 0
        for batch in self.train_loader:
            with torch.no_grad():
                out_dict = self.model(batch)
                flat_codes = []
                for code_feat in self.code_feats:
                    flat_codes.append(out_dict[code_feat].long().flatten() + self.min_val)
                batch_bins = torch.bincount(torch.cat(flat_codes))
                if len(batch_bins) > len(self.bins):
                    batch_bins[: len(self.bins)] += self.bins
                    self.bins = batch_bins
                elif len(self.bins) > len(batch_bins):
                    self.bins[: len(batch_bins)] += batch_bins
                else:
                    self.bins += batch_bins
            num_images += len(batch)
            if num_images > self.num_train_images:
                break

        bins = self.bins.float()
        bins_smooth = (
                    (bins + alpha) / (bins.sum() + len(bins) * alpha)).cpu()  # additive smooth counts using alpha
        self.cdf = rc.prob_to_cum_freq(bins_smooth, resolution=2 * len(bins_smooth))  # convert pdf -> cdf
        self.model.train()

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
        flat_code_list = []
        for x in code_list:
            # Holds each flattened featured to be concatenated after the loop
            assert x.shape[0] == 1, "can only compress one image at a time, for now"
            flat_code_list.append(x.long().flatten())
        full_code = torch.cat(flat_code_list) + self.min_val
        # use RangeEncoder to write compressed representation to file
        encoder = rc.RangeEncoder(savepath)
        code_list = full_code.tolist()
        encoder.encode(code_list, self.cdf)
        encoder.close()

