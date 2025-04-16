save_dir=results/output_step1
mkdir $save_dir


# datafile=./data/mimic_cxr/mimic_annotation.json
datafile=./data/mimic_cxr/longitudinal_mimic_annotation.json

images_path=/path/to/mimic-cxr-jpg/images/

# stage 1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=12360 main_train.py --image_dir $images_path --ann_path $datafile --model_name encoder_decoder --dataset_name mimic_cxr --image_encoder resnet101 --gen_max_len 150 --gen_min_len 100 --batch_size 16 --epochs 6 --save_dir $save_dir --seed 456789 --init_lr 5e-5 --min_lr 5e-6 --warmup_lr 5e-7 --weight_decay 0.05 --warmup_steps 2000 --cls_weight 1 --con_weight 1  2>&1 | tee "$save_dir/log_train.log" 
