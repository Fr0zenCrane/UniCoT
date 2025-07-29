gpu_num=8

for i in $(seq 0 $((gpu_num-1)));
do
    CUDA_VISIBLE_DEVICES=$i python inference_mdp_self_reflection_wise.py \
        --group_id $i \
        --group_num $gpu_num \
        --model_path "Fr0zencr4nE/UniCoT-7B-MoT" \
        --data_path "./eval/gen/wise/final_data.json" \
        --outdir "./results" \
        --cfg_text_scale 4 > process_log_$i.log 2>&1 &
done

wait
echo "All background processes finished."