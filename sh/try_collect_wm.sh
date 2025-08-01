# python collect_wm_text.py --wm_name "SynthID" --dataset_name "../../dataset/c4/realnewslike" --model_name "../model/Llama3.1-8B_hg" --file_num 50 --file_data_num 100
# python collect_wm_text.py --wm_name "DIP" --dataset_name "../../dataset/c4/realnewslike" --model_name "../model/Llama3.1-8B_hg" --file_num 1 --file_data_num 1
# python collect_wm_text.py --wm_name "EXPGumbel" --dataset_name "../../dataset/c4/realnewslike" --model_name "../model/Llama3.1-8B_hg" --file_num 1 --file_data_num 1
# python collect_wm_text.py --wm_name "Unigram" --dataset_name "../../dataset/c4/realnewslike" --model_name "../model/Llama3.1-8B_hg" --file_num 1 --file_data_num 1
# python collect_wm_text.py --wm_name "Unbiased" --dataset_name "../../dataset/c4/realnewslike" --model_name "../model/Llama3.1-8B_hg" --file_num 50 --file_data_num 100
# python collect_wm_text.py --wm_name "SIR" --dataset_name "../../dataset/c4/realnewslike" --model_name "facebook/opt-1.3b" --file_num 1 --file_data_num 1
# python collect_wm_text.py --wm_name "UPV" --dataset_name "../../dataset/c4/realnewslike" --model_name "facebook/opt-1.3b" --file_num 1 --file_data_num 1


python collect_wm_text.py --wm_name "KGW" --dataset_name "../../dataset/c4/realnewslike" --model_name "facebook/opt-1.3b" --file_num 2 --file_data_num 100 --device 0
python collect_wm_text.py --wm_name "SynthID" --dataset_name "../../dataset/c4/realnewslike" --model_name "../model/Llama3.1-8B_hg" --file_num 50 --file_data_num 100 --device 0 --language 'fr'