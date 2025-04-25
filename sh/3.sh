# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std 3 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True"




python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --max_edit_rate 0.13

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "SynthID" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --max_edit_rate 0.13

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --max_edit_rate 0.13

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "DIP" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --max_edit_rate 0.13

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unbiased" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --max_edit_rate 0.13