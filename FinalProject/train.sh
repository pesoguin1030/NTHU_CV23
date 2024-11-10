echo "start training"
CUDA_VISIBLE_DEVICES=0 \
python main.py --model_name CSPPeleeNet_small \
--model_save_dir results/CSPPeleeNet_small/weights/ \
--mode train \
--num_epoch 50 \
--batch_size 36 \
--data_dir /local/SSD1/CV_Final_Group6/dataset/ \


echo "train pass ^^"
