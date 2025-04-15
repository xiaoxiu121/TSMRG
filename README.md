# TSMRG

Code of paper: "TSMRG: Temporal Semantic Enhancement for Medical Report Generation with Longitudinal Data".

## Installation
1. Create a new conda environment.
```Shell
conda create -n tsmrg python=3.10
conda activate tsmrg
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```
## Datasets Preparation
* **MIMIC-CXR**: The images can be downloaded from [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/), and put it under `data/mimic_cxr/images`.  First, the annotation file `mimic_annotation.json` is fully aligned with the [HERGen](https://github.com/HKU-MedAI/HERGen), ensuring identical entry counts and dataset structure. In addition to the standard entries, we have extended the annotation file by adding the following fields:  

     * `labels` and `clip_indices`: These are essential for classification tasks and are derived from [PromptRMG](https://github.com/jhb86253817/PromptMRG).
     * `report_comparision`: This field contains a temporally-aware report summary, generated using the [InternLM2-20B](https://huggingface.co/internlm/internlm2-20b) model.





* **Longitudinal-MIMIC-CXR**: It's a subset of MIMIC-CXR. The file `longitudinal_mimic_annotation.json` is generated using the same procedure as described above.

Simultaneously, we integrate the `clip_text_features.json` and  `base_probs.json`from [PromptRMG](https://github.com/jhb86253817/PromptMRG) and put it under `data/mimic_cxr/`, which is needed for classification module.

Moreover, you need to download the `chexbert.pth` from [here](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) for evaluating clinical efficacy and put it under `checkpoints/stanford/chexbert/`.

You will have the following structure:
````
TSMRG
|--data
   |--mimic_cxr
      |--base_probs.json
      |--clip_text_features.json
      |--mimic_annotation.json
      |--longitudinal_mimic_annotation.json
      |--images
         |--p10
         |--p11
         ...
         ...
|--checkpoints
   |--stanford
      |--chexbert
         |--chexbert.pth
...
````

## Training
**stage 1**: Training using individual image-text pairs
```
bash train_mimic_cxr_step1.sh
```

**stage 2**: Training the model using additional contrastive alignment
```
bash train_mimic_cxr_step2.sh
```
Please change the `ckpt_path` to the pretrained model path in stage 1.


## Testing
Run `bash test_mimic_cxr.sh` to test a trained model on MIMIC-CXR or Longitudinal-MIMIC-CXR

## Acknowledgment

The code was adapted and inspired by the fantastic works of [PromptMRG](https://github.com/jhb86253817/PromptMRG) and  [HERGen](https://github.com/HKU-MedAI/HERGen).
