
import gensim.downloader
from tqdm import tqdm
import numpy as np
from textattack.utils import Logger, to_string, truncation
import datetime
from copy import deepcopy

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
        
        self.gensimi=GensimModel()
        self.tokenizer = tokenizer
        self.vocab_size=self.tokenizer.vocab_size
        self.token_len_flag=True
        self.simi_num_for_token=5

        
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

    def wm_wipe(
        self, sentence, 
        max_edit_rate=0.3,
    ):

        token_ids=self.tokenizer.encode(sentence)
        # token_ids=remove_repeat(token_ids)
        token_list=[self.tokenizer.decode(t, skip_special_tokens=True) for t in token_ids]
        token_num=len(token_list)
        
        rand_tokens=np.random.choice(token_num, int(token_num*max_edit_rate), replace=False)

        edit_dist=0
        new_token_list=deepcopy(token_list)
        for token_id in rand_tokens:
            tmp_token=token_list[token_id]
            tmp_subst=self.substitute(tmp_token)
            if len(tmp_subst)>0:
                new_token_list[token_id]=tmp_subst[0]
                edit_dist+=1

        new_sentence=''.join(new_token_list)

        return new_sentence, edit_dist