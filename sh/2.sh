python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std 3 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std 4 --device 1 --remove_spoof "True"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --max_edit_rate 0.13


python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --max_edit_rate 0.13