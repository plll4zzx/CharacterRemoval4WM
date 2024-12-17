import gzip
import json
import os
import numpy as np
from textattack.shared.utils import device, words_from_text

def count_loc(text, words, word_num):
    if word_num>=len(words):
        word_num=-1
        return len(text)
    word0=words[max(word_num-1,0)]
    word=words[word_num]
    word1=words[min(word_num+1, len(words)-1)]
    word_len=len(word)
    word_len0=len(word0)
    word_len1=len(word1)
    text_len=len(text)
    for idx in range(text_len-1, 0, -1):
        tmp=text[idx-word_len:idx]
        tmp0=text[idx-word_len-word_len0-3:idx-word_len]
        tmp1=text[idx:min(idx+word_len1+3, text_len)]
        if tmp==word and tmp0.find(word0)>-1 and tmp1.find(word1)>-1:
            return min(idx, int(word_num*10))
    return int(word_num*10)

class c4:

    def __init__(self, dir_path, file_num=50, file_data_num=10, rand_seed=123):
        self.dir_path=dir_path
        self.file_num=int(file_num)
        self.file_data_num=int(file_data_num)
        self.data_num=int(file_num*file_data_num)
        self.all_file_list=np.array(os.listdir(self.dir_path))
        # np.random.randint(0, len(self.file_list), self.file_num)
        np.random.seed(rand_seed)
        self.file_index=np.random.choice(len(self.all_file_list), self.file_num).tolist()
        self.file_list=self.all_file_list[self.file_index].tolist()
        
        self.data=[]
    
    def load_data(self, text_len=50):
        for file_name in self.file_list:
            file_path=os.path.join(self.dir_path, file_name)
            json_file = gzip.open(file_path, 'rb')
            # json_list = json_file.readlines()
            # data_index = np.random.choice(len(json_list), self.file_data_num).tolist()
            counter=0
            while counter<self.file_data_num:
                data_json = json_file.readline()
                data_dict = json.loads(data_json)
                tmp_text=data_dict['text']
                words=words_from_text(tmp_text)
                char_loc=count_loc(tmp_text, words, text_len)
                self.data.append((tmp_text[0:char_loc],0))
                counter+=1

# def parse(path):
#     g = gzip.open(path, 'rb')
#     for l in g:
#         yield json.loads(l)

# def get_data(dir_path, file_name):
#     file_path=os.path.join(dir_path, file_name)
#     for d in parse(file_path):
#         print(d)

if __name__=='__main__':
    dir_path='/home/plll/dataset/c4/realnewslike'
    # file_name='c4-train.00000-of-00512.json.gz'
    # get_data(dir_path, file_name)
    c4_dataset=c4(dir_path=dir_path)
    c4_dataset.load_data()
    print()