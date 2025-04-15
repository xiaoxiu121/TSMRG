import torch
import torch.nn as nn
import torchvision.models as models

import os
from torch.nn import Linear, Module
from typing import Optional

from .cvt_default import _C as config
from .cvt_default import _update_config_from_file
from.cls_cvt import get_cls_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../")


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def spatial_position_feature_size(model_config):
    sizes = {
        "cvt-13-224x224": 384,
        "cvt-13-384x384": 384,
        "cvt-21-224x224": 384,
        "cvt-21-384x384": 384,
        "cvt-w24-384x384": 1024,
    }
    return sizes[model_config]


class CvT(Module):
    """
    CvT implementation from https://github.com/leoxiaobin/CvT.
    """

    def __init__(
            self,
            warm_start: bool,
            model_config: Optional[str] = None,
            ckpt_name: Optional[str] = None,
            num_classes: Optional[int] = None,
            is_encoder: bool = False,
            freeze_domain_params: bool = False,
            ckpt_path: Optional[str] = None,
            **kwargs,
    ):
        """
        Argument/s:
            warm_start - warm-start with checkpoint.
            model_config - model configuration for CvT,
            ckpt_name - name of the checkpoint for the model.
            num_classes - number of classes for replacement head.
            is_encoder - if the network is being used as an encoder.
            freeze_domain_params - freeze domain-specific parameters.
            implementation_version - implementation version.
            ckpt_path - path of the pre-trained model checkpoints.
            kwargs - keyword arguments.
        """
        super(CvT, self).__init__()

        self.warm_start = warm_start
        self.num_classes = num_classes
        self.is_encoder = is_encoder

        # CvT
        args = Namespace(
            cfg=os.path.join(REPO_ROOT_DIR, "configs", "cvt", model_config + ".yaml"))
        _update_config_from_file(config, args.cfg)
        self.cvt = get_cls_model(config)
        if self.warm_start:
            checkpoint = torch.load(ckpt_path,
                                    map_location=torch.device(
                                        'cpu') if not torch.cuda.is_available() else None,
                                    )
            self.cvt.load_state_dict(checkpoint)

        # Classification head
        if num_classes:
            self.cvt.head = Linear(
                config["MODEL"]["SPEC"]["DIM_EMBED"][-1], num_classes)

        # Freeze domain-specific parameters (transfer learning)
        if freeze_domain_params:
            for n, p in self.cvt.named_parameters():
                p.requires_grad = False

    def forward(self, images: torch.FloatTensor):
        """
        Forward propagation.

        Argument/s:
            images (torch.Tensor) - a batch of images.

        Returns
            Dictionary of outputs.
        """
        for i in range(self.cvt.num_stages):
            images, cls_tokens = getattr(self.cvt, f'stage{i}')(images)
        outputs = {}
        if self.is_encoder:
            outputs["last_hidden_state"] = torch.flatten(images, start_dim=2)
        if self.num_classes:
            outputs["logits"] = self.cvt.head(
                torch.squeeze(self.cvt.norm(cls_tokens)))
        return outputs


class blip_cvt(nn.Module):
    def __init__(self, args):
        super(blip_cvt, self).__init__()
        # Load from the original repo
        self.encoder = CvT(
            warm_start=True,
            model_config='cvt-21-384x384',
            ckpt_name='CvT-21-384x384-IN-22k',
            ckpt_path=os.path.join(
                REPO_ROOT_DIR, "checkpoints/cvt/CvT-21-384x384-IN-22k.pth"),
            is_encoder=True
        )
        emb_dim = 384

        self.encoder_compact = nn.Conv1d(
            in_channels=576, out_channels=49, kernel_size=1)
        self.encoder_projection = nn.Linear(emb_dim, 768)
        
        map_size = 7
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=map_size, stride=1, padding=0)
        
    def forward(self, images):
        
        image_features = self.encoder(images)['last_hidden_state'] # 图像要用指定的transformers
        image_features = image_features.permute(0, 2, 1)
        image_features = self.encoder_compact(image_features) # [B,576,384]->[B,49,384]
        
        # compact the image features
        image_features = self.encoder_projection(image_features) # ->[B,49,768]
        
        # avg
        patch_feats = image_features.permute(0, 2, 1)
        batch_size, feat_size, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, 7, 7 )
        
        avg_feats = self.avg_fnt(patch_feats).flatten(1)

        
        return image_features, avg_feats
    
    
        
