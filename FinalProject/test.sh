echo "start training"
CUDA_VISIBLE_DEVICES=0 \
python main.py --model_name model_b1 \
--model_save_dir results/origin_efficientb1/weights \
--mode test \
--result_dir results/origin_efficientb1/ \
--data_dir /local/SSD1/CV_Final_Group6/dataset/ \

echo "test pass ^^"
