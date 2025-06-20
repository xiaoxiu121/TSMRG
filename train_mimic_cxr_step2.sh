save_dir=results/output_step2
mkdir $save_dir


# datafile=./data/mimic_cxr/mimic_annotation.json
datafile=./data/mimic_cxr/longitudinal_mimic_annotation.json

images_path=/path/to/mimic-cxr-jpg/images/

ckpt_path=/path/to/model_step1.pth

# stage 2
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 --master_port=12361 main_train.py --image_dir $images_path --ann_path $datafile --model_name temporal_model --dataset_name mimic_cxr --image_encoder resnet101 --load_pretrained $ckpt_path --gen_max_len 150 --gen_min_len 100 --batch_size 16 --epochs 6 --save_dir $save_dir --seed 456789 --init_lr 5e-5 --min_lr 5e-6 --warmup_lr 5e-7 --weight_decay 0.05 --warmup_steps 200 --cls_weight 4 --con_weight 0.5 2>&1 | tee "$save_dir/log_train.log" 
