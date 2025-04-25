# import random
import numpy as np
from pygad import GA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from llm_wm import LLM_WM
from textattack.utils import Logger, to_string, save_json, save_jsonl, truncation, find_homo, random_keyboard_neighbor
import datetime
from copy import copy
from difflib import SequenceMatcher
from random_attack import GensimModel
from text_OCR import text_OCR_text
import random
from spellchecker import SpellChecker
import Levenshtein
import language_tool_python

zwsp = chr(0x200B)
zwj = chr(0x200D)
ZWNJ=chr(0x200C)
WJ=chr(0x2060)
VS16=chr(0xFE0F)

class GA_Attack:
    def __init__(
        self, 
        victim_model = 'bert-base-uncased', 
        victim_tokenizer = 'bert-base-uncased',
        wm_detector=None,
        logger=None,
        wm_name='TS',
        fitness_threshold=0.90,
        device='cuda',
        len_weight=1.3,
        eva_thr=0.2,
        mean=0,
        std=1,
        ab_std=1,
        atk_style='char',
        ori_flag=False,
        ocr_flag=False,
        def_stl='',
    ):
        self.gensimi=None 
        self.simi_num_for_token=5
        self.tokenizer = AutoTokenizer.from_pretrained(victim_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(victim_model)
        self.model.eval()
        self.device=device
        self.special_char = [zwsp, ZWNJ]#[' ', chr(0x200B), chr(0xFE0F)]#
        self.wm_detector=wm_detector
        self.wm_name=wm_name
        self.fitness_threshold=fitness_threshold
        self.best_solution=None
        self.best_fitness=0
        self.edit_distance=0
        self.best_sentence=''
        self.len_weight=len_weight
        self.eva_thr=eva_thr
        self.mean=mean
        self.std=std
        self.ab_std=ab_std
        self.atk_style=atk_style
        self.ori_flag=ori_flag
        self.ocr_flag=ocr_flag
        self.def_stl=def_stl

        if ocr_flag:
            # self.spellchecker = SpellChecker()
            self.spellchecker = language_tool_python.LanguageTool('en-US')

        self.model.to(self.device)

        if logger is None:
            self.log=Logger(
                'attack_log/GAAttack'+'-'.join([
                    self.wm_name, 
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
        if isinstance(info, str):
            self.log.logger.info(info)
        elif isinstance(info, dict):
            keys='\t'.join([key for key in info])
            values='\t'.join([str(info[key]) for key in info])
            self.log.logger.info(keys)
            self.log.logger.info(values)
        else:
            info=to_string(info)
            self.log.logger.info(info)

    def truncation(self, text, max_token_num=100):
        new_text, token_num=truncation(
            text, 
            # self.tokenizer, 
            max_token_num=max_token_num
        )
        return new_text, token_num
    
    def get_abnormal_tokens(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        
        input_ids = inputs["input_ids"]         # shape: [1, seq_length]
        attention_mask = inputs["attention_mask"]
        
        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        embeddings.requires_grad_()
        embeddings.retain_grad()
        
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        
        score = outputs.logits  
        print(f"Predicted score: {score.item():.4f}")
        
        
        self.model.zero_grad()  
        score.backward()
        
        
        gradients = embeddings.grad.detach()  
        
        token_grad_norms = gradients.norm(dim=-1).squeeze(0) 

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        mean_grad = token_grad_norms.mean().item()
        std_grad = token_grad_norms.std().item()
        
        threshold = mean_grad + self.ab_std * std_grad

        abnormal_tokens = []
        for token, norm in zip(tokens, token_grad_norms.cpu().numpy()):
            if norm > threshold:
                abnormal_tokens.append(token)

        return abnormal_tokens
    
    def check_abnormal_token(self, token):
        for a_token in self.abnormal_tokens:
            similarity = SequenceMatcher(None, token, a_token).ratio()
            if similarity>0.5:
                return True
        return False   

    def evaluate_fitness(self, modified_sentence, target_class):
        if self.ori_flag:
            return (self.wm_detector(modified_sentence)['score']-self.mean)/self.std
        
        inputs = self.tokenizer(modified_sentence, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        fitness=logits
        
        return fitness.item()
    
    def substitute(self, token):
        if self.gensimi is None:
            self.gensimi=GensimModel()
        space_flag=False
        if ' ' in token:
            space_flag=True
        token=token.replace(' ', '').lower()
        # if self.token_len_flag:
        if len(token)<=2:
            return []
        
        if token in self.simi_tokens_dict:
            return self.simi_tokens_dict[token]
        simi_tokens=self.gensimi.find_simi_words(token, simi_num=self.simi_num_for_token)

        if space_flag:
            for idx in range(len(simi_tokens)):
                if simi_tokens[idx][0]!=' ':
                    simi_tokens[idx]=' '+simi_tokens[idx]
        self.simi_tokens_dict[token]=simi_tokens
        
        return simi_tokens
    
    def modify_sentence(self, solution):
        edited_sentence = list(self.tokens)
        selected_tokens = []

        for i, gene in enumerate(solution):
            if gene > 0 and len(selected_tokens) < self.max_edits:  # Enforce max edits
                if self.check_abnormal_token(edited_sentence[i]):
                    continue
                len_t=len(edited_sentence[i])
                sep_len=3
                m_num=len_t//sep_len
                if m_num==0:
                    continue
                m_locs=[int(len_t//2)]
                # m_locs=[int((m_num/len_t)*(j)*len_t)+(sep_len//2) for j in range(m_num)]
                # half_token_len=len(edited_sentence[i])//2
                # if half_token_len<=1:
                #     continue

                tmp_token=copy(edited_sentence[i])

                if self.atk_style=='token':
                    tmp_subst=self.substitute(tmp_token)
                    if len(tmp_subst)>0:
                        edited_sentence[i]=tmp_subst[0]
                        selected_tokens.append(i)
                    continue

                # # Treat operation as part of the solution
                # operation = random.choice([2,2, 4, 5])
                # operation=gene
                operation = 2 
                for m_loc in m_locs:
                    if m_loc>=(len_t-1):
                        continue
                    
                    tmp_char=tmp_token[m_loc]
                    if self.ocr_flag:
                        if tmp_token in self.best_adv_token:
                            tmp_token=self.best_adv_token[tmp_token]
                            continue
                        tmp_token_list=[
                            tmp_token[:m_loc] + find_homo(tmp_char)+ ' ' + random_keyboard_neighbor(tmp_token[m_loc+1])+ tmp_token[m_loc+2:], #2 4
                            # tmp_token[:m_loc] + tmp_token[m_loc+1]+find_homo(tmp_char)+ tmp_token[m_loc+2:], #2 4
                            # tmp_token[:m_loc] + find_homo(random_keyboard_neighbor(tmp_char))+ tmp_token[m_loc+1:], #2 5
                            # tmp_token[:m_loc]+ ' ' + find_homo(tmp_char)+ tmp_token[m_loc+1:],
                        ]
                        # w=[1,3,1]
                        dist_dict={}
                        for idx, word in enumerate(tmp_token_list):
                            # c_word=''#self.spellchecker.correction(word)
                            if 'spell' in self.def_stl:
                                matches = self.spellchecker.check(word)
                                c_word=language_tool_python.utils.correct(word, matches)
                                if c_word is None:
                                    c_word=word
                            else:
                                c_word=text_OCR_text(word, style='ocr_t')
                                cid=c_word.find('\n')
                                if cid>-1:
                                    c_word=c_word[:cid]
                            c_word=c_word.lower()
                            word=word.lower()
                            d0=Levenshtein.distance(word, c_word)
                            d1=Levenshtein.distance(word, tmp_token)
                            d2=Levenshtein.distance(c_word, tmp_token)
                            dc=d1*0.5+d2
                            dist_dict[idx]=(dc, tmp_token, word, c_word, d0, d1, d2)
                        sorted_keys_desc = sorted(dist_dict.keys(), key=lambda k: dist_dict[k][0], reverse=True)#
                        self.best_adv_token[tmp_token]=tmp_token_list[sorted_keys_desc[0]]
                        tmp_token=tmp_token_list[sorted_keys_desc[0]]#
                        continue
                    if operation == 1:  # Delete
                        tmp_token = tmp_token[:m_loc] + tmp_token[m_loc+1:]
                    elif operation == 2:  # Replace
                        tmp_token = tmp_token[:m_loc] +find_homo(tmp_char)+ tmp_token[m_loc+1:] 
                    elif operation == 3:  # Insert
                        tmp_token = tmp_token[:m_loc] + random.choice(self.special_char)+ tmp_token[m_loc:]
                    elif operation == 4:  # swap
                        tmp_token = tmp_token[:m_loc] + tmp_token[m_loc+1]+tmp_token[m_loc]+ tmp_token[m_loc+2:]
                    elif operation == 5:  # typo
                        tmp_token = tmp_token[:m_loc] + random_keyboard_neighbor(tmp_char)+ tmp_token[m_loc+1:]
                edited_sentence[i]=tmp_token
                
                selected_tokens.append(i)
        # Reconstruct sentence
        modified_sentence = " ".join(edited_sentence)
        edit_distance=len(selected_tokens)
        return modified_sentence, edit_distance, np.abs(edit_distance-self.max_edits)

    def fitness_function(self, ga_instance, solution, solution_idx):
        
        modified_sentence, solu_len, _ = self.modify_sentence(solution)
        # if self.ocr_flag:
        #     modified_sentence=text_OCR_text(modified_sentence)

        # Evaluate fitness using the helper function
        eva_fitness=-self.evaluate_fitness(modified_sentence, self.target_class)
        
        if eva_fitness<self.eva_thr and self.ori_flag==False:
            fit_score = eva_fitness
        else:
            fit_score = eva_fitness+(self.max_edit_rate-solu_len/solution.size)*self.len_weight
        # elif eva_fitness>=self.fitness_threshold:
        #     fit_score = self.fitness_threshold+(-solu_len/solution.size)*self.len_weight
        # if eva_fitness>self.eva_thr:
        #     fit_score=eva_fitness-(solu_len/solution.size)*self.len_weight+(self.max_edit_rate)*self.len_weight
        if self.ori_flag==False:
            if abs(fit_score-self.best_fitness)<0.05:
                return min(fit_score, self.best_fitness-0.0001)#
            if abs(solu_len-self.edit_distance)>=2 and self.edit_distance>0:
                return min(fit_score, self.best_fitness-0.0001)#
        if self.remove_spoof:
            return fit_score
        else:
            return -fit_score

    def get_adv(
        self, sentence, target_class, ori_fitness,
        num_generations=30, 
        num_parents_mating=50, 
        population_size=100, 
        max_edit_rate=0.1, 
        mutation_percent_genes=30,
        remove_spoof=True, #remove=True spoof=False
    ):

        # sentence=sentence.lower()
        # tmp_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
        # sentence=self.tokenizer.decode(tmp_ids, skip_special_tokens=True)
        
        # if self.ocr_flag:
        #     population_size=20
        #     mutation_percent_genes=50#5
        #     num_parents_mating=10
        
        self.remove_spoof=remove_spoof
        self.tokens = sentence.split()
        self.ori_fitness=ori_fitness
        self.target_class = target_class
        self.max_edit_rate=max_edit_rate

        # if self.ori_fitness>1.5:
        #     self.max_edit_rate+=0.05

        self.max_edits = max(1, int(np.ceil(len(self.tokens) * self.max_edit_rate))) 
        n = len(self.tokens)

        self.best_solution=None
        self.best_sentence=''
        self.edit_distance=0
        self.best_fitness=-100
        self.abnormal_tokens=[]
        if self.ab_std>0:
            self.abnormal_tokens=self.get_abnormal_tokens(sentence)
        self.simi_tokens_dict={}

        if self.ocr_flag:
            self.best_adv_token={}

        # Initialize PyGAD
        ga_instance = GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=self.fitness_function,
            sol_per_pop=population_size,
            num_genes=n,
            gene_type=int,
            init_range_low=0,
            init_range_high=2,  # Operations have three possible values: 0, 1, 2, 3
            mutation_percent_genes=mutation_percent_genes,
            on_generation=self.on_generation,
        )

        # Run the algorithm
        ga_instance.run()

        # Output the best solution
        # best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        # if best_solution!=self.best_solution:
        #     print()
        # best_sentence, edit_distance, solu_len = self.modify_sentence(best_solution)
        adv_fitness=self.evaluate_fitness(self.best_sentence, self.target_class)
        self.log_info(['adv_fitness:', adv_fitness])
        return self.best_sentence, self.edit_distance, self.best_fitness

    def on_generation(self, ga_instance):
        # self.log_info(f"Generation: {ga_instance.generations_completed}")
        # steps=ga_instance.generations_completed
        # self.best_solution=best_solution
        (best_solution, best_fitness, _)=ga_instance.best_solution()
        self.log_info(f"Generation: {ga_instance.generations_completed}, Best Fitness: {best_fitness}")
        if self.wm_detector is not None:
            best_sentence, edit_distance, _ = self.modify_sentence(best_solution)
            wm_rlt=self.wm_detector(best_sentence)
            self.log_info(f"el_detect: {wm_rlt}, edit_distance: {edit_distance}")
            self.log_info(f"tp_sentence: {best_sentence}")

        if best_fitness>self.best_fitness:
            self.best_fitness=best_fitness
            self.best_solution=best_solution
            self.best_sentence=best_sentence
            self.edit_distance=edit_distance
        if edit_distance <= self.max_edits*0.5:
            self.log_info(f"Stop! Edit_distance reached.")
            return "stop"
        # if steps%5==0 and steps>0:
        #     self.abnormal_tokens=self.get_abnormal_tokens(self.best_sentence)
        # if best_fitness >= (self.fitness_threshold+(self.max_edit_rate)*self.len_weight*0.5):
        #     self.log_info(f"Stop! Fitness threshold reached.")
        #     return "stop"
        

# Example usage
if __name__ == "__main__":
    llm_name="facebook/opt-1.3b"
    wm_name="KGW"
    ga_attack = GA_Attack(
        victim_model = 'saved_model/RefDetector_KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2024-12-23',
        victim_tokenizer = 'bert-base-uncased',
        wm_detector=LLM_WM(model_name = llm_name, device = "cuda", wm_name=wm_name).detect_wm
    )
    sentence = "attempt at promotion to english football ’ s second tier. \" having been relegated from the championship ( south preston had struggled at the wrong end ) to this campaign and fighting back through promotion gives us great confidence, \" said lewer – a league one winner with barnsley in 1997, twice a northern irish premier league winner with hamilton and with northampton town, as well as a top - flight player. sign up to our daily newsletter the i newsletter cut through the noise sign up thanks for signing up! sorry, there seem to be some issues. please try again later. submitting... blackpool manager alan lewier gives instructions to his team from the technical area during blackpool v rotherham united match at bloomfield road, blackpool, saturday may 29, 2021. ( photo by tony johnson ). the seasiders kicked off the new league one season with a 1 - 0 success at rotherham, but blackpool's 3 - 3 draw with oldham, with"
    # sentence = ""

    target_class = 1  # Target class index to maximize
    ori_fitness=ga_attack.evaluate_fitness(sentence, target_class)
    ori_wm_rlt=ga_attack.wm_detector(sentence)
    ga_attack.log_info(["Original fitness:", ori_fitness])
    ga_attack.log_info(["Original WM Detect:", ori_wm_rlt])
    best_sentence, edit_distance, best_fitness = ga_attack.get_adv(
        sentence, target_class,
        max_edit_rate=0.15,
        num_generations=30,
    )
    ga_attack.log_info(["Best solution:", best_sentence])
    ga_attack.log_info(["edit_distance:", edit_distance])
    ga_attack.log_info(["Best fitness:", best_fitness])
