
import gensim.downloader
from tqdm import tqdm
import numpy as np
from textattack.utils import Logger, to_string, truncation
import datetime
from copy import deepcopy
import string

class GensimModel:

    def __init__(self):
        self.model_name_list=[
            'glove-twitter-25',
        ]
        self.low_model_list=[
            'glove-twitter-25',
            'glove-wiki-gigaword-50',
        ]
        self.high_model_list=[
            'word2vec-google-news-300',
            # 'glove-twitter-100',
            # 'glove-wiki-gigaword-100',
        ]
        self.model_dict={
            model_name:gensim.downloader.load(model_name)
            for model_name in self.model_name_list
        }
        self.simi_words_dict={
            model_name:{}
            for model_name in self.model_name_list
        }

    def find_simi_words(self, target_word, simi_num=10):
        found_words=[]
        words_scores=[]
        for model_name in self.model_name_list:
            try:
                if target_word in self.simi_words_dict[model_name]:
                    similar_words=self.simi_words_dict[model_name][target_word]
                else:
                    similar_words = self.model_dict[model_name].most_similar(target_word)
                    self.simi_words_dict[model_name][target_word]=similar_words
                for (word, score) in similar_words:
                    if word not in found_words:
                        found_words.append(word)
                        words_scores.append((word, score))
                    if len(found_words)>simi_num:
                        break
            except:
                # print(model_name+' do not find')
                continue
        if len(found_words)>simi_num:
            found_words=found_words[0:simi_num]
        return found_words


class RandomAttack:
    def __init__(
        self,
        tokenizer,
        logger=None,
    ):
        
        self.gensimi=None 
        self.tokenizer = tokenizer
        self.vocab_size=self.tokenizer.vocab_size
        self.token_len_flag=True
        self.simi_num_for_token=5
        self.special_char=string.whitespace
        
        if logger is None:
            self.log=Logger(
                'attack_log/RandomAttack'+'-'.join([
                    # self.wm_name, self.attack_name, 
                    # self.victim_name.replace('/','_'), self.llm_name.replace('/','_'),
                    # str(self.temperature)
                ])+'-'+str(datetime.datetime.now())[0:-10]+'.log',
                level='debug', 
                screen=False
            )
        else:
            self.log=logger
        self.log_info('\n')

    def log_info(self, info=''):
        if not isinstance(info, str):
            info=to_string(info)
        self.log.logger.info(info)

    def truncation(self, text, max_token_num=100):
        new_text, token_num=truncation(text, self.tokenizer, max_token_num)
        return new_text, token_num

    def substitute(self, token):
        if self.gensimi is None:
            self.gensimi=GensimModel()
        space_flag=False
        if ' ' in token:
            space_flag=True
        token=token.replace(' ', '').lower()
        if self.token_len_flag:
            if len(token)<=2:
                return []
            # if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
            #     return []
        # simi_token_ids=[]
        simi_tokens=self.gensimi.find_simi_words(token, simi_num=self.simi_num_for_token)

        if space_flag:
            for idx in range(len(simi_tokens)):
                if simi_tokens[idx][0]!=' ':
                    simi_tokens[idx]=' '+simi_tokens[idx]
        
        return simi_tokens

    def token_attack(
        self, sentence, 
        max_edit_rate=0.3,
    ):
        token_ids=self.tokenizer.encode(sentence)
        # token_ids=remove_repeat(token_ids)
        token_list=[self.tokenizer.decode(t, skip_special_tokens=True) for t in token_ids]
        token_num=len(token_list)
        
        max_edit_dist=round(token_num*max_edit_rate)
        rand_tokens=np.random.choice(token_num, max_edit_dist*3, replace=False)

        edit_dist=0
        new_token_list=deepcopy(token_list)
        for token_id in rand_tokens:
            tmp_token=token_list[token_id]
            tmp_subst=self.substitute(tmp_token)
            if len(tmp_subst)>0:
                new_token_list[token_id]=tmp_subst[0]
                edit_dist+=1
            # else:
            #     print()
            if edit_dist>=max_edit_dist:
                break

        new_sentence=''.join(new_token_list)

        return new_sentence, edit_dist
    
    def char_attack(
        self, sentence, 
        max_edit_rate=0.3,
    ):
        token_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
        token_ids_dict={idx:t for idx, t in enumerate(token_ids)}
        token_dict={idx:self.tokenizer.decode(t, skip_special_tokens=True) for idx, t in enumerate(token_ids)}
        token_num=len(token_ids)
        
        max_edit_dist=round(token_num*max_edit_rate)
        rand_tokens=np.random.choice(token_num, min(token_num, max_edit_dist*3), replace=False)
        rand_opt=np.random.randint(0, 3, min(token_num, max_edit_dist*3))

        edit_dist=0
        new_token_ids_dict=deepcopy(token_ids_dict)
        for ids, token_idx in enumerate(rand_tokens):
            tmp_token=token_dict[token_idx]
            if len(tmp_token)<5:
                continue
            tmp_opt=rand_opt[ids]

            rand_idx=np.random.randint(2, len(tmp_token)-1)
            rand_char_id=np.random.choice(len(self.special_char))
            rand_char=self.special_char[rand_char_id]
            if tmp_opt==0: # delete
                tmp_token=tmp_token[0:rand_idx]+tmp_token[rand_idx+1:]
            elif tmp_opt==1: # insert
                tmp_token=tmp_token[0:rand_idx]+rand_char+tmp_token[rand_idx:]
            elif tmp_opt==2: # substitute
                tmp_token=tmp_token[0:rand_idx]+rand_char+tmp_token[rand_idx+1:]

            new_token_ids_dict[token_idx]=self.tokenizer.encode(tmp_token, add_special_tokens=False)
            edit_dist+=1
            if edit_dist>=max_edit_dist:
                break
        new_token_ids=[]
        for idx in new_token_ids_dict:
            if isinstance(new_token_ids_dict[idx], list):
                new_token_ids+=new_token_ids_dict[idx]
            else:
                new_token_ids.append(new_token_ids_dict[idx])
        new_sentence=self.tokenizer.decode(new_token_ids)
        return new_sentence, edit_dist 