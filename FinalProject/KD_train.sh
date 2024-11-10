echo "start training CSP_KD_v2_h0.0_s1.0"
CUDA_VISIBLE_DEVICES=0 \
python KD_simple_v2_train.py \
--model_save_dir results/CSP_KD_v2_h0.0_s1.0/weights \
--hard_loss_ratio 0.0 \
--soft_loss_ratio 1.0 \
--data_dir /local/SSD1/CV_Final_Group6/dataset/ \


# echo "start training CSP_KD_v1_h0.0_s1.0"
# CUDA_VISIBLE_DEVICES=0 \
# python KD_simple_train.py \
# --model_save_dir results/CSP_KD_v1_h0.0_s1.0/weights \
# --hard_loss_ratio 0.0 \
# --soft_loss_ratio 1.0 \
# --data_dir /local/SSD1/CV_Final_Group6/dataset/ \

# echo "start training CSP_KD_v3_h0.8_s0.2"
# CUDA_VISIBLE_DEVICES=0 \
# python KD_simple_v3_train.py \
# --model_save_dir results/CSP_KD_v3_h0.8_s0.2/weights \
# --hard_loss_ratio 0.8 \
# --soft_loss_ratio 0.2\
# --data_dir /local/SSD1/CV_Final_Group6/dataset/ \


