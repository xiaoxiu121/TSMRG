import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption
import os
import pandas as pd
from datasets import Dataset as Dataset_hg



CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


class generation(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, split='train', max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation[split]
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        with open('./data/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
    
        # create huggingface Dataset class, which is easy to tokenize
        dataset_as_dfs = pd.DataFrame(self.ann)

        # FIXME: remove this line later
        # longitudinal mimic
        self.min_seq_length = 2
        self.max_seq_length = 5
        subject_cnts = dataset_as_dfs["subject_id"].value_counts()
        cur_subject_cnts = subject_cnts[subject_cnts >= self.min_seq_length]
        dataset_as_dfs = dataset_as_dfs.loc[dataset_as_dfs["subject_id"].isin(
            cur_subject_cnts.index.tolist())]

        dataset = Dataset_hg.from_pandas(dataset_as_dfs)

        subject_cnts = dataset_as_dfs["subject_id"].value_counts()
        cur_subject_cnts = subject_cnts[subject_cnts >= self.min_seq_length]
        dataset_as_dfs = dataset_as_dfs.loc[dataset_as_dfs["subject_id"].isin(
            cur_subject_cnts.index.tolist())]

        subject_cnts = dataset_as_dfs["subject_id"].value_counts()

        subject_cnts = subject_cnts[subject_cnts > self.max_seq_length]
        dataset_part_1 = dataset_as_dfs.loc[~dataset_as_dfs["subject_id"].isin(
            subject_cnts.index.tolist())]
        dataset_part_2 = dataset_as_dfs.loc[dataset_as_dfs["subject_id"].isin(
            subject_cnts.index.tolist())]

        # we split all patient sequence which have more than 5 images into several sequences
        if len(dataset_part_2) > 0:
            new_dfs = []
            for subject_id in dataset_part_2["subject_id"].unique():
                sub_df = dataset_part_2.loc[dataset_part_2["subject_id"]
                                            == subject_id]
                sub_df.sort_values(
                    by=["subject_id", "StudyDate", "StudyTime"], inplace=True)

                num_images = len(sub_df)
                for i in range(num_images // self.max_seq_length + 1):
                    cur_sub_df = sub_df.iloc[i *
                                             self.max_seq_length: (i + 1) * self.max_seq_length]
                    cur_sub_df["subject_id"] = str(subject_id) + "_" + str(i)
                    new_dfs.append(cur_sub_df)

            dataset_part_2 = pd.concat(new_dfs, axis=0)
            self.dataset_as_dfs = pd.concat(
                [dataset_part_1, dataset_part_2], axis=0)
        else:
            self.dataset_as_dfs = dataset_part_1
        self.dataset_as_dfs.reset_index(inplace=True, drop=True)

        self.unique_ptids = self.dataset_as_dfs["subject_id"].unique().tolist()
        self.dataset_as_dfs["subject_id"].value_counts()
        
        
    def __len__(self):
        return len(self.unique_ptids)
    
    def __getitem__(self, index):    
        
        ptid = self.unique_ptids[index]
        sub_df = self.dataset_as_dfs.loc[self.dataset_as_dfs["subject_id"] == ptid]
        sub_df.sort_values(by=["StudyDate", "StudyTime"], inplace=True)
        sub_df.reset_index(inplace=True, drop=True)

        id_list, image_list, report_list, study_date_list, cls_labels_list, prompt_list, clip_memory_list, reportcom_list  = [
        ], [], [], [], [], [], [], []
        for example in sub_df.to_dict(orient="records"):
            image_path = example["image_path"][0]
            image = Image.open(os.path.join(self.image_root, image_path)).convert('RGB')
            image = self.transform(image)
            image_list.append(image)
            
            clip_indices = example['clip_indices'][:self.args.clip_k] 
            clip_memory = self.clip_features[clip_indices]
            clip_memory = torch.from_numpy(clip_memory).float()
            clip_memory_list.append(clip_memory)
            
            cls_labels_list.append(torch.from_numpy(np.array(example['labels'])).long())
            
            prompt = [SCORES[l] for l in example['labels']]
            prompt_list.append(' '.join(prompt)+' ')
            report_list.append(my_pre_caption(example['report'], self.max_words))
            reportcom_list.append(my_pre_caption(example['report_comparision'], self.max_words))
            id_list.append(example["id"])
            study_date_list.append(example["StudyDate"])

        image_list = torch.stack(image_list)
        clip_memory_list = torch.stack(clip_memory_list) # 5,21,512
        
        cls_labels_list = torch.stack(cls_labels_list)

        # now we need to pad samples to make sure they can appear in the same batch
        batch_mask = torch.tensor([False] * self.max_seq_length)
        batch_mask[:len(id_list)] = True

        if len(id_list) < self.max_seq_length:
            # pad several nan samples
            pad_length = self.max_seq_length - len(id_list)
            # id_list.extend([""] * pad_length)
            report_list.extend([""] * pad_length)
            reportcom_list.extend([""] * pad_length)
            prompt_list.extend([""] * pad_length)

            pad_value = study_date_list[0]
            study_date_list.extend([pad_value] * pad_length)

            # pad images
            _, c, h, w = image_list.shape
            pad_image = torch.zeros(pad_length, c, h, w)
            image_list = torch.cat([image_list, pad_image], dim=0)
            
            _, l,d  = clip_memory_list.shape
            pad_clip_memory = torch.zeros(pad_length, l,d)
            clip_memory_list = torch.cat([clip_memory_list, pad_clip_memory], dim=0)
            
            pad_cls_label = torch.zeros(pad_length, cls_labels_list.shape[-1]).long()
            cls_labels_list = torch.cat([cls_labels_list, pad_cls_label], dim=0)

        study_date_list = torch.tensor(study_date_list).float()
        pad_value = study_date_list[0]
        study_date_list = study_date_list - pad_value
        study_date_list = study_date_list / 1000.
        
        return image_list, report_list, reportcom_list, batch_mask, study_date_list, cls_labels_list, prompt_list, clip_memory_list


    
