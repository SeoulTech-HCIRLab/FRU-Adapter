#scripts/mafw/main.sh
finetune_dataset='mafw'
num_labels=11
input_size=224
sr=4
# splits=("1" "2" "3" "4" "5") # you can change it to other folds, e.g., (2,3,4,5)
splits=(5)
# splits=(1)
lr=1e-4 # 8e-4
min_lr=1e-5 #8e-6
epochs=100 # 300
# model_checkpoint="./saved/model/finetuning/${finetune_dataset}/MAFW_org/eval_split0${splits}/checkpoint-best.pth" #

for split in "${splits[@]}";
    do
      OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/MAFW_org_train/eval_split0${split}"
      if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p $OUTPUT_DIR
      fi
      # path to split files (train.csv/val.csv/test.csv)
      DATA_PATH="./saved/data/${finetune_dataset}/single/split0${split}"
      # batch_size can be adjusted according to number of GPUs
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
          --master_port 43210 \
          main_org.py \
          --data_set ${finetune_dataset^^} \
          --nb_classes ${num_labels} \
          --data_path ${DATA_PATH} \
          --log_dir ${OUTPUT_DIR} \
          --output_dir ${OUTPUT_DIR} \
          --batch_size 4 \
          --num_sample 1 \
          --input_size ${input_size} \
          --short_side_size ${input_size} \
          --save_ckpt_freq 1000 \
          --num_frames 16 \
          --sampling_rate ${sr} \
          --opt adamw \
          --lr ${lr} \
          --opt_betas 0.9 0.999 \
          --epochs ${epochs} \
          --test_num_segment 2 \
          --test_num_crop 2 \
          --num_workers 16 \
          --weight_decay 0.05 \
          --smoothing 0.1 \
          --min_lr ${min_lr} \
          --warmup_lr ${min_lr} \
          #--eval ${model_checkpoint}\
          #--finetune ${model_checkpoint}\

done
echo "Done!"


