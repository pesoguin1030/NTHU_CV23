可執行的檔案 有4個
	- train.sh
		用來 direct train model 
	- test.sh
		用來 inference direct model 然後跑出csv
	- KD_train.sh
		可以選擇要執行 $python KD_simple_train.py (v1)/ KD_simple_v2_train.py (v2)/ KD_simple_v3_train.py (v3)
	- KD_test.py
		用來 inference KD model (也就是壓縮過後的student model)
		
	**執行的時候，記得要改 --data_dir 路徑**	
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------


以下將解說各個 檔案 有什麼比較特別的參數可以設定

train.sh
	- setting args (in .sh)
		- model_save_dir
			train好參數存的地方
		- model_name
			選擇direct train 的model
		- data_dir
			資料路徑
-----------------------	
test.sh
	- setting args (in .sh)
		- model_save_dir
			選擇要讀取的weight檔案
		- model_name
			選擇direct train 的model
		- result_dir
			inference出來的csv要放在哪
		- data_dir
			資料路徑
-----------------------	

KD_train.sh
	python KD_simple_xx_train.py
	+ KD_simple_train.py
		- setting args (in .sh)
			- model_save_dir 
				train好參數存的地方
			- hard_loss_ratio = 0.8
				這個是和GT label的crossEntropy loss
			- soft_loss_ratio = 0.2
				這個是和teacher所predict的label比的crossEntropy loss
		- setting directly in .py
			- data_dir
				training data path
			- T = 10
			- loss_mse_ratio = 1
				這個是teacher和student最後一層feature map的mse loss
				
	+ KD_simple_v2_train.py
		- setting args (in .sh)
			- model_save_dir 
				train好參數存的地方
			- hard_loss_ratio = 0.8
				這個是和GT label的crossEntropy loss
			- soft_loss_ratio = 0.2
				這個是和teacher所predict的label比的crossEntropy loss
		- setting directly in .py
			- data_dir
				training data path
			- T = 10
			- loss_mse_ratio = 1
				這個是teacher和student最後一層feature map的mse loss
			- each_features_ratio
				每一層layer's feature map的mse loss ratio
				
	+ KD_simple_v3_train.py
		- setting args (in .sh)
			- model_save_dir 
				train好參數存的地方
			- hard_loss_ratio = 0.8
				這個是和GT label的crossEntropy loss
			- soft_loss_ratio = 0.2
				這個是和teacher所predict的label比的crossEntropy loss
		- setting directly in .py
			- data_dir
				training data path
			- T = 10
			- loss_mse_ratio = 1
				這個是teacher和student最後一層feature map的mse loss
			- each_features_ratio
				每一層layer's feature map的mse loss ratio
-----------------------	
				
KD_test.py			
	- setting directly in .py
		- KD_model_weight_path 
			train好的KD_model's weight路徑
		- result_dir
			inference出來的csv要放在哪
		- data_dir
			data 路徑
		- student_model_name
		- KD_model_name