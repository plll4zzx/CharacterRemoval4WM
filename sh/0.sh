python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 10
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 0 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15
python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 5 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 15  --do_flag "False"

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "DIP" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 15  --do_flag "False"

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "SynthID" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 15  --do_flag "False"

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 5  --do_flag "False"

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "Unbiased" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 15  --do_flag "False"

# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5 
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 10
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 15 

# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "DIP" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5 --do_flag "False"
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "DIP" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 10 --do_flag "False"
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "DIP" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 --do_flag "False"

# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "SynthID" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5 --do_flag "False"
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "SynthID" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 10
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "SynthID" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0 --remove_spoof "True" --num_generations 15 

# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5 
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 10
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unigram" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 

# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unbiased" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 5 
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unbiased" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 10
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "Unbiased" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -2 --device 0 --remove_spoof "True" --num_generations 15 