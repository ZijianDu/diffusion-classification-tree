#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"


## ddim100: 100 is stride, will run 100 times
SAMPLE_FLAGS="--batch_size 16 --num_samples 16 --selected_class 245 --timestep_respacing ddim1000 --num_diffusion_samples 1000 --save_results True 
--output_path results/ --plots_dir plots/ --image_data_path datasets/ --random_crop_per_image 4 --random_h_flip 0.5 --random_v_flip 0.5"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 
--image_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 
--num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

python classifier_sample.py $MODEL_FLAGS --classifier_scale 0.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS