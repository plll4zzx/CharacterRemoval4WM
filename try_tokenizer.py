import tiktoken
from transformers import AutoTokenizer

tokenizer_type='bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
# tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 的 BPE 词表
text_l=['language', "lanɡuage", "lanuage","large lan𝐠uage model"]
for text in text_l:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_list=[tokenizer.decode(id, skip_special_tokens=True) for id in token_ids]
    print(token_ids)  # 输出 token ID 序列
    print(token_list)  # 还原 token 对应的文本
    print(tokenizer.decode(token_ids, skip_special_tokens=True))
