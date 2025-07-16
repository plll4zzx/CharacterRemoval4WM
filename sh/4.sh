python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "SynthIDfr" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std 1 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "True" --ocr_flag "False" --max_edit_rate 0.23

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGWfr" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std 1 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "True" --ocr_flag "False" --max_edit_rate 0.23

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "DIP" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" --max_edit_rate 0.11
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "SynthID" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" --ab_std 0 --def_stl "ocr"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" --max_edit_rate 0.21  --ab_std 1
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unbiased" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" --ab_std 1


python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  --ab_std 2 --def_stl "ocr"
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "DIP" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  --ab_std 2  --def_stl "ocr"
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "SynthID" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  --ab_std 3  --def_stl "ocr"
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  --ab_std 2  --max_edit_rate 0.21 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unbiased" --atk_style "token" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  --ab_std 3


python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "DIP" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "SynthID" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False" 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unbiased" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15   --do_flag "False"  

python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "char" --ori_flag "True" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5   --do_flag "False" --max_edit_rate 0.1 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "DIP" --atk_style "char" --ori_flag "True" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5   --do_flag "False"  --max_edit_rate 0.1 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "SynthID" --atk_style "char" --ori_flag "True" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5   --do_flag "False"  --max_edit_rate 0.1 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "True" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5   --do_flag "False"   --max_edit_rate 0.1 
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unbiased" --atk_style "char" --ori_flag "True" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5   --do_flag "False"  --max_edit_rate 0.1 

python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --def_stl "del"
python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --ocr_flag "True" --def_stl "del"