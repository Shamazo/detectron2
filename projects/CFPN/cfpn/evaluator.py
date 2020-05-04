from detectron2.evaluation import DatasetEvaluator
import torch
import torch.nn.functional as F
import detectron2.data.transforms as T
from detectron2.data import build_detection_test_loader
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from collections import defaultdict
from typing import List, Dict
from scipy.stats import entropy
import numpy as np
import range_coder as rc
import os
import copy
import io
import matplotlib.pyplot as plt
import seaborn as sbn
from PIL import Image

from detectron2.utils.events import get_event_storage


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
        orig_image = torch.unsqueeze(orig_image, dim=0).float().to(outputs[0][self.eval_img].get_device())

        reconstruct_image = outputs[0][self.eval_img].float()
        reconstruct_image = reconstruct_image[:, :, 0:orig_image.shape[2], 0:orig_image.shape[3]]
        assert orig_image.shape[1] == 3, "original image must have 3 channels"
        assert reconstruct_image.shape[1] == 3, "reconstructed image must have 3 channels"
        with torch.no_grad():
            ssim_val = ssim(reconstruct_image, orig_image, data_range=255, size_average=False)
            ms_ssim_val = ms_ssim(reconstruct_image, orig_image, data_range=255, size_average=False)
            self.ssim_vals.extend(ssim_val)
            self.ms_ssim_vals.extend(ms_ssim_val)

    def evaluate(self):
        mean_ssim = torch.mean(torch.stack(self.ssim_vals)).cpu().item()
        mean_ms_ssim = torch.mean(torch.stack(self.ms_ssim_vals)).cpu().item()
        print({'image-sim': {"ssim": mean_ssim, "ms-ssim": mean_ms_ssim}})
        return {'image-sim': {"ssim": mean_ssim, "ms-ssim": mean_ms_ssim}}


class CompressionEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name, model=None):
        """
        Args:
            model: model to be evaluated.
        """
        self.kodak_cfg = copy.deepcopy(cfg)
        # TODO if we want to evaluate compression on other datasets we need a new config key
        self.test_loader = build_detection_test_loader(self.kodak_cfg, 'kodak_test')
        self.model = model.eval()
        self.code_feats = cfg.MODEL.QUANTIZER.IN_FEATURES
        self.num_train_images = cfg.TEST.NUM_COMPRESSION_IMAGES
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.output_dir = cfg.OUTPUT_DIR
        self.negative_codes = cfg.TEST.NEGATIVE_CODES
        return

    def reset(self):
        self.cdf = None
        self.pdf = dict()
         # number of pixels in an image, does not include padding.
        self.test_pixels = dict()
         # pdf bins
        self.bins = dict()
        for feat in self.code_feats:
            self.bins[feat] = torch.tensor([0.0]).to(self.device).long()
        self.processed_images = 0
        return

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a compressive model
            outputs: the outputs of a compressive model given those inputs, must
                contain the keys listed in cfg.MODEL.QUANTIZER.IN_FEATURES
        """
        if self.processed_images > 5:
            return
        flat_codes = []
        for _, output in zip(inputs, outputs):
            self.processed_images += 1
            # for code_feat in self.code_feats:
            #     flat_codes.append((output[code_feat].clone().detach().long().flatten()))
            # fc_len = [fc.shape[0] for fc in flat_codes]
            # fc_len = sum(fc_len)
            # fc_tensor = torch.zeros([fc_len], device=self.device, dtype=torch.long)
            # idx = 0
            # print("concat")
            batch_bins = dict()
            for feat in self.code_feats:
                batch_bins[feat] = torch.bincount(output[feat].long().flatten())
            for feat in self.code_feats:
                if len(batch_bins[feat]) > len(self.bins[feat]):
                    batch_bins[feat][: len(self.bins[feat])] += self.bins[feat]
                    self.bins[feat] = batch_bins[feat]
                elif len(self.bins[feat]) > len(batch_bins[feat]):
                    self.bins[feat][: len(batch_bins[feat])] += batch_bins[feat]
                else:
                    self.bins[feat] += batch_bins[feat]

        return

    def evaluate(self):
        """
        First we build the latent distribution on a subset of the training set
        then we compress our testset, writing to file and then we see how many bits
        the file is on disk, returning an average.
        Returns:

        """
        self.build_latent_distribution()
        self.visualize_pdf()
        bpp = []
        # both of these are lists of dicts for each input, with feat keys
        entropy_lst = []
        rel_entropy_lst = []
        seen_kodak_images = 0
        for inputs in self.test_loader:
            outputs = self.model(inputs)

            key = "img_{}".format(seen_kodak_images)
            self.test_pixels[key] = inputs[0]['image'].shape[1] * inputs[0]['image'].shape[2]
            seen_kodak_images += 1
            savepath = os.path.join(self.output_dir, key + ".dci")
            size, entropy, rel_entropy = self.compress_image(outputs[0], savepath)
            if size: # size = 0 if encoder fails
                bpp.append(size / self.test_pixels[key])
            entropy_lst.append(entropy)
            rel_entropy_lst.append(rel_entropy)


        self.model.train()
        metric_dict = {"bpp": np.mean(bpp)}
        for feat in self.code_feats:
            metric_dict['{} entropy'.format(feat)] = np.mean([x[feat] for x in entropy_lst])
            metric_dict['{} rel_entropy'.format(feat)] = np.mean([x[feat] for x in rel_entropy_lst])
            print("REL ENTROPY", metric_dict['{} rel_entropy'.format(feat)])
        return {'Compression': metric_dict}

    def visualize_pdf(self, alpha=1):
        storage = get_event_storage()
        for feat in self.code_feats:
            plt.clf()
            fig = plt.figure()
            np_bins = self.pdf[feat].cpu().numpy()
            plt.plot(np.arange(0, len(np_bins)), np_bins)
            plt.xlabel("Code Value")
            plt.ylabel("Probability")
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt_image = np.array(Image.open(buf))[:, :, :-1]  # drop transparent channel
            plt_name = "PDF for {}".format(feat)
            storage.put_image(plt_name, plt_image.transpose(-1, 0, 1)) # put image is channel first
            plt.close('all')
        return

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
        # num_images = 0
        # self.model.eval()
        # if self.negative_codes:
        #     for batch in self.train_loader:
        #         with torch.no_grad():
        #             out_dict = self.model(batch)
        #             for code_feat in self.code_feats:
        #                 self.min_val = torch.min(
        #                     self.min_val,
        #                     out_dict[code_feat].long().min(),
        #                 )
        #         num_images += len(batch)
        #         if num_images > self.num_train_images:
        #             break

        # self.min_val = self.min_val.abs()
        # self.bins = torch.tensor([0.0]).to(self.device).long()
        # num_images = 0
        # for batch in self.train_loader:
        #     with torch.no_grad():
        #         print(num_images)
        #         out_dict = self.model(batch)
        #         flat_codes = []
        #         for code_feat in self.code_feats:
        #             flat_codes.append(out_dict[code_feat].long().flatten() + self.min_val)
        #         batch_bins = torch.bincount(torch.cat(flat_codes))
        #         if len(batch_bins) > len(self.bins):
        #             batch_bins[: len(self.bins)] += self.bins
        #             self.bins = batch_bins
        #         elif len(self.bins) > len(batch_bins):
        #             self.bins[: len(batch_bins)] += batch_bins
        #         else:
        #             self.bins += batch_bins
        #     num_images += len(batch)
        #     if num_images > self.num_train_images:
        #         break

        for feat in self.code_feats:
            bins = self.bins[feat].float()
            bins_smooth = (
                        (bins + alpha) / (bins.sum() + len(bins) * alpha)).cpu()  # additive smooth counts using alpha
            self.pdf[feat] = bins_smooth
            self.cdf[feat] = rc.prob_to_cum_freq(bins_smooth, resolution=2 * len(bins_smooth))  # convert pdf -> cdf
        # self.model.train()

    def compress_image(self, codes: Dict[str, torch.Tensor], savepath="default.dci"):
        """
        Args:
            codes: dict [str->tensor] mapping code name to code tensor
            savepath: path to save the image to
        Returns: size of files on disk, also writes compressed image savepath
        """

        if not self.cdf:
            raise Exception(
                "Cannot compress image without cdf function (hint: call build_latent_distribution)"
            )
        flat_code_dict = dict()
        size = 0
        entropy_dict = dict()
        rel_entropy = dict()
        failed_encode = False # if the encoder fails due to mismatch in cdf sizes
        for feat in self.code_feats:
            assert len(codes[feat].shape) == 1, "Code feats must be flat. Shape: {}".format(codes[feat].shape)
            segment_path = savepath[:-4] + feat +".dci"
            encoder = rc.RangeEncoder(segment_path)
            bin_count = torch.bincount(codes[feat].long()).float()
            image_pdf = (bin_count / bin_count.sum()).cpu().numpy()
            pdf = self.pdf[feat].cpu().numpy()
            # scipy entropy requires the inputs for relative entropy to be the same shape
            len_pdf = len(pdf)
            len_image_pdf = len(image_pdf)
            # print("{} img bin count: {}".format(feat, bin_count))
            # print("{} img pdf: {}".format(feat, image_pdf))
            if len_pdf > len_image_pdf:
                image_pdf = np.pad(image_pdf, (0, len_pdf - len_image_pdf))
            elif len_image_pdf > len_pdf:
                pdf = np.pad(pdf, (0, len_image_pdf - len_pdf))
            # if the pdf is len 1 the features are all zeros, scipy would return nan due to divide by 0
            if len_image_pdf == 1:
                entropy_dict[feat] = 0
            else:
                entropy_dict[feat] = entropy(image_pdf, base=2)

            rel_entropy[feat] = entropy(image_pdf, qk=pdf, base=2)
            # print(rel_entropy[feat])
            # print(image_pdf.shape)
            # print(pdf.shape)
            # print(image_pdf)
            # print(pdf)
            try:
                encoder.encode(codes[feat].long().tolist(), self.cdf[feat])
                encoder.close()
                size += os.stat(segment_path).st_size * 8
            except:
                print("ValueError: An entry in `data` is too large or `cumFreq` is too short.")
                failed_encode = True
        if failed_encode:
            size = 0
        return size, entropy_dict, rel_entropy

