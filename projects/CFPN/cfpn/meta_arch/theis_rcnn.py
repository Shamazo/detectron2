import logging
from detectron2.config import CfgNode
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import torch
import cv2
import numpy as np

from ..reconstruct_heads.reconstruct_heads import build_reconstruct_heads
from ..backbone import CompressiveEncoderBackbone # import so they are included in the registry!
from ..reconstruct_heads import CompressiveDecoderHead


@META_ARCH_REGISTRY.register()
class RCNNwithReconstruction(GeneralizedRCNN):
    def __init__(self, cfg: CfgNode):
        super(RCNNwithReconstruction, self).__init__(cfg)
        self.reconstruct_heads = build_reconstruct_heads(cfg, self.backbone.output_shape()).to(self.device)

    def forward(self, batched_inputs: torch.Tensor):
        if not self.training:
            return self.inference(batched_inputs, )
        images_to_compare = self.preprocess_image(batched_inputs, norm=False)
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        reconstructed_images, reconstruction_losses = self.reconstruct_heads(images_to_compare, features)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(reconstructed_images, images)
                self.visualize_proposals(batched_inputs, proposals, results=None)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(reconstruction_losses)
        return losses

    def inference(self, batched_inputs, **kwargs):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            Dict[str->tensor] mapping output image names to the tensor in n,c,h,w format
            :param batched_inputs:
            :param **kwargs:
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor.float())

        reconstructed_images, loss_dict = self.reconstruct_heads(images, features)
        return reconstructed_images

    def preprocess_image(self, batched_inputs: torch.Tensor, norm=True):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if norm:
            images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def visualize_proposals(self, batched_inputs, proposals, results=None, threshold=0.90):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 5
        for input, prop in zip(batched_inputs, proposals):
            temp = (prop.objectness_logits.sigmoid()<threshold).nonzero()
            if len(temp) > 0:
                max_vis_prop = min(max_vis_prop, temp[0])
                
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            if results:
                v_pred = v_pred.overlay_instances(
                    boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy(),
                    labels=results[0].gt_classes[0:box_size]
                )
            else:
                v_pred = v_pred.overlay_instances(
                    boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
                )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def visualize_training(self, reconstructed_images, images):
        storage = get_event_storage()
        orig_img = images.tensor[0].long().detach().cpu().numpy()
        if self.input_format == "BGR":
            orig_img = orig_img[::-1, :, :]
        for key in reconstructed_images:
            recon_img = reconstructed_images[key][0].long().detach().cpu().numpy()
            assert orig_img.shape[0] == 3, "Images should have 3 channels."
            assert recon_img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                recon_img = recon_img[::-1, :, :]
            _, height, width = recon_img.shape


            t_orig_img = orig_img.transpose(1, 2, 0).astype("uint8")
            ds_orig_img = cv2.resize(t_orig_img, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
            recon_img = recon_img.transpose(1, 2, 0).astype("uint8")
            # print(ds_orig_img.shape, recon_img.shape)
            vis_img = np.concatenate((ds_orig_img, recon_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "{} ;Left: GT image;  Right: reconstructed image".format(key)
            storage.put_image(vis_name, vis_img) #takes in c,h,w images
        return