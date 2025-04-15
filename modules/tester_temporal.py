import logging
import os
from abc import abstractmethod
import numpy as np
import time

import cv2
import torch

from .metrics_clinical import CheXbertMetrics

class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader

    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        all_num= 0
        cls_true_num= 0
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, captions, _, batch_masks, study_date, cls_labels, _, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                clip_memory = clip_memory.to(self.device) 
                reports, cls_preds, _ = self.model.generate(images, clip_memory, batch_masks, study_date, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
                cls_labels = cls_labels[batch_masks]
                B, T, C, H, W = images.shape
                ground_truths = []
                for j in range(B):
                    for i in range(T):
                        c = captions[i][j]
                        if len(c)>0:ground_truths.append(c)
                
                # 计算预测准确率
                cls_p = np.array(cls_preds) # b,18
                cls_gt = cls_labels.numpy() # b,14

                cls_p = cls_p[:,:cls_gt.shape[-1]]
                result= (cls_p == cls_gt).astype(int)     
                
                all_num += cls_gt.shape[0]*cls_gt.shape[1]
                cls_true_num+=result.sum()

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                if batch_idx % 10 == 0:
                    print('{}/{}'.format(batch_idx, len(self.test_dataloader)))
                    
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
            log.update(**{'test_' + 'cls_acc': cls_true_num/all_num})
        return log

