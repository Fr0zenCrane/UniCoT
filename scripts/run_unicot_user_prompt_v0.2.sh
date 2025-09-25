gpu_num=8

for i in $(seq 0 7);
do
    CUDA_VISIBLE_DEVICES=$i python inference_unicot_v0.2.py \
        --group_id $i \
        --group_num $gpu_num \
        --mode pure_breakdown \
        --model_path "ByteDance-Seed/BAGEL" \
        --model_fine_tuned_path "path/to/unicot" \
        --data_path "./test_prompts.txt" \
        --output_dir "./results/" \
        --cfg_text_scale 4 \
        --cfg_img_scale 2.0 \
        > process_log_$i.log 2>&1 &
done

wait
echo "All background processes finished."
