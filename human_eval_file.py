import pandas as pd
from textattack.utils import load_json
import random

def read_text_json(file_path, text_type):
    data_list=load_json(file_path=file_path)
    res_list=[]
    for data_dict in data_list:
        if data_dict['wm_detect']['is_watermarked']==True and data_dict['adv_detect']['is_watermarked']==False:
            if text_type=='wm':
                res_list.append({
                    'text':data_dict['wm_text'],
                    'type':text_type,
                    'rand':random.randint(0,1000)
                })
            else:
                res_list.append({
                    'text':data_dict['adv_text'],
                    'type':text_type,
                    'rand':random.randint(0,1000)
                })
        if len(res_list)>50:
            break
    return res_list


# 假设你的数据如下：
data_wm = read_text_json(
    file_path='saved_attk_data/Rand_0.1_100_sentence_1_False__RefDetector_KGW_.._.._dataset_c4_realnewslike_.._model_Llama3.1-8B_hg_bert-base-uncased_2025-02-13.json', 
    text_type='wm'
)
data_token = read_text_json(
    file_path='saved_attk_data/Rand_0.1_100_token_10_False__RefDetector_KGW_.._.._dataset_c4_realnewslike_.._model_Llama3.1-8B_hg_bert-base-uncased_0.15_9.json', 
    text_type='token')
data_char = read_text_json(
    file_path='saved_attk_data/Rand_0.1_100_char_10_False__RefDetector_KGW_.._.._dataset_c4_realnewslike_.._model_Llama3.1-8B_hg_bert-base-uncased_0.15_9.json', 
    text_type='char')
data_sen = read_text_json(
    file_path='saved_attk_data/Rand_0.1_100_sentence_1_False__RefDetector_KGW_.._.._dataset_c4_realnewslike_.._model_Llama3.1-8B_hg_bert-base-uncased_2025-02-13.json', 
    text_type='sen'
)
data=data_wm+data_token+data_char+data_sen
# 将 list[dict] 转为 DataFrame
df = pd.DataFrame(data)

# 保存为 Excel 文件（第一行自动是列名）
df.to_excel("human_eval.xlsx", index=False)