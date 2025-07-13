import tiktoken
from transformers import AutoTokenizer
from textattack.utils import homos, save_json

zwsp = chr(0xFE0F)
tokenizer_type='../model/Llama3.1-8B_hg'#'facebook/opt-1.3b'#
tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
# tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 的 BPE 词表
text_l=['lang'+zwsp+'uage','lanɡuage', "lanɡuage", "lanuage","large lanɡuage model"]
for text in text_l:
# homo_dict=[]
# for c1 in homos:
    # text = homos[c1]
    # token_ids = {
    #     str(id)+'_'+t:tokenizer.encode(t, add_special_tokens=False)
    #     for (id,t) in enumerate(text)
    # }
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_list=[tokenizer.decode(id, skip_special_tokens=True) for id in token_ids]
    print(text)  # 输出 token ID 序列
    print(token_ids)  # 输出 token ID 序列
    print(token_list)  # 还原 token 对应的文本
    # print(tokenizer.decode(token_ids, skip_special_tokens=True))
#     homo_dict.append(token_ids)
# save_json(homo_dict, 'homo_opt.json')
# from openai import OpenAI
# client = OpenAI()

# response = client.moderations.create(
#     model="omni-moderation-latest",
#     input="the world’s people, a reporter ask𝐞d: “W𝐡y n𐐬t t𝐡e saｍe for t𝐡e Palestiｎians?” Bus𝐡’s resp𐐬nse was th𝐚t “th𝐞re is no suⲥh thｉng as a Palestinian. Th𝐞re aⲅe Israelis aｎd Palestinians.” This is a signi𝚏icant comｍent siｎce it is oｎe of t𝐡e fiⲅst times Buꮪh h𝐚s publicly maᏧe t𝐡e point th𝐚t he h𝐚s maᏧe privately to Isr𝐚eli lead𝐞rs. Wh𝐞n he met wi𝚝h Mahｍoud Abbas last May 26, Bush asked hiｍ, “W𝐡at do y𐐬u caᛁl yourself?” AbᏏas respoｎded, “I’m a Palestｉnian.” Buꮪh said, “I don’t kn𐐬w anybody who calls himself a Palestinian. I kn𐐬w Israelis aｎd Palestinians.” T𝐡e Israeli newsⲣaper Haaⲅetz repoⲅted",
# )

# print(response)