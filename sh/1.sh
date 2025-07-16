
# python train_ref_detector.py --wm_name "Unigram"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 9 --device 1
# python train_ref_detector.py --wm_name "Unigram"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 5 --device 1 
# python train_ref_detector.py --wm_name "Unigram"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 0 --device 1
# python train_ref_detector.py --wm_name "KGW"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 9 --device 1
# python train_ref_detector.py --wm_name "KGW"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 5 --device 1 
# python train_ref_detector.py --wm_name "KGW"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 0 --device 1
# python train_ref_detector.py --wm_name "Unigram"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "facebook/opt-1.3b" --ths 4     --rand_times 9 --device 1
# python train_ref_detector.py --wm_name "Unigram"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "facebook/opt-1.3b" --ths 4     --rand_times 5 --device 1 
# python train_ref_detector.py --wm_name "Unigram"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "facebook/opt-1.3b" --ths 4     --rand_times 0 --device 1
python train_ref_detector.py --wm_name "SynthIDfr"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 0.52    --rand_times 0 --device 0
python train_ref_detector.py --wm_name "SynthIDfr"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 0.52    --rand_times 9 --device 0
python train_ref_detector.py --wm_name "SynthIDfr"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 0.52    --rand_times 5 --device 0
python train_ref_detector.py --wm_name "KGWfr"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 9 --device 0
python train_ref_detector.py --wm_name "KGWfr"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 5 --device 0
python train_ref_detector.py --wm_name "KGWfr"      --num_epochs 15 --rand_char_rate 0.15 --llm_name "../model/Llama3.1-8B_hg" --ths 4     --rand_times 0 --device 0

python test_rand_sh.py --llm_name "facebook/opt-1.3b" --wm_name_list "['KGW']" --atk_style_list "['token','char']" --ori_flag "False" --data_aug_list "[0]" --max_edit_rate_list "[0.25]" --do_flag "True" --atk_times_list "[1]" --max_token_num_list "[100]"

python train_ref_detector.py --wm_name "KGW" --num_epochs 15 --rand_char_rate 0.15 --llm_name "facebook/opt-1.3b" --ths 4  --rand_times 0 --device 0

python test_rand_sh.py --llm_name "facebook/opt-1.3b" --wm_name_list "['KGW']" --atk_style_list "['sand_char']" --ori_flag "False" --data_aug_list "[9]" --max_edit_rate_list "[0.25]" --do_flag "True" --atk_times_list "[1]" --max_token_num_list "[100]"

python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "KGW" --atk_style "char" --ori_flag "False" --data_aug 9 --device 0 --num_generations 15  --do_flag "True" --max_edit_rate 0.13 