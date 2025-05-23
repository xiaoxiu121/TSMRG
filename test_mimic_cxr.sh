save_dir=results/output_step2
datafile=./data/mimic_cxr/mimic_annotation.json

ckpt_path=/path/to/model.pth
images_path=/path/to/mimic-cxr-jpg/images/


CUDA_VISIBLE_DEVICES=1, python main_test.py \
--n_gpu 1 \
--image_dir $images_path \
--ann_path $datafile \
--model_name temporal_model \
--gen_max_len 150 \
--gen_min_len 100 \
--batch_size 16 \
--save_dir $save_dir \
--seed 456789 \
--load_pretrained $ckpt_path \
2>&1 | tee "$save_dir/log_test.log" 

