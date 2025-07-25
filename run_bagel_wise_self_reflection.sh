gpu_num=8

for i in $(seq 0 7);
do
    CUDA_VISIBLE_DEVICES=$i python distributed_grouped_inference_wise_mdp_enhance_mdp_loop.py \
        --group_id $i \
        --group_num $gpu_num \
        --model_path "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/qlz/pretrained_models/BAGEL-7B-MoT" \
        --model_fine_tuned_path "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/qlz/projects/unified_vidgen_blip3o/BagelCoT/results_mdp_v1_mr_fft/checkpoints/0003000" \
        --data_path "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/gongjia/datasets/bagel_output/sampled-wise-0712" \
        --cfg_text_scale 4 > process_log_$i.log 2>&1 &
done

wait
echo "All background processes finished."