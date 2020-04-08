from detectron2.evaluation import DatasetEvaluator
import torch
import torch.nn.functional as F
import detectron2.data.transforms as T
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class CompressionEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, eval_img="img_2"):
        """
        Args:
            dataset_name: must be 'kodak_test' for now
            output_dir:
            eval_img: the key in the output which contains the image we are evaluating
                Currently hard coding to img_2 which is the largest, but in the future
                we could also compare the lower resolution reproductions
        """
        assert dataset_name=='kodak_test', "Can only evaluate compression on kodak_test"
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
        assert(len(inputs) == 1)
        #reshape the input image to have a maximum length of 512 as the model preprocesses
        #Much shuffling of data, but also the dataset is only 24 images
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