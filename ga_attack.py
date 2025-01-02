import random
import numpy as np
from pygad import GA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from llm_wm import LLM_WM
from textattack.utils import Logger, to_string, save_json, save_jsonl, truncation, find_homo
import datetime

class GA_Attack:
    def __init__(
        self, 
        victim_model = 'bert-base-uncased', 
        victim_tokenizer = 'bert-base-uncased',
        wm_detector=None,
        logger=None,
        wm_name='TS',
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(victim_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(victim_model)
        self.special_char = '@'
        self.wm_detector=wm_detector
        self.wm_name=wm_name

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
        if not isinstance(info, str):
            info=to_string(info)
        self.log.logger.info(info)

    def truncation(self, text, max_token_num=100):
        new_text, token_num=truncation(text, self.tokenizer, max_token_num)
        return new_text, token_num

    def evaluate_fitness(self, modified_sentence, target_class, solu_len=0):
        # Tokenize the modified sentence
        inputs = self.tokenizer(modified_sentence, return_tensors="pt", padding=True, truncation=True)

        # Evaluate fitness using the fine-tuned classification model
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)

        # Define a fitness value based on the target misclassification
        fitness = predictions[0][target_class].item()

        return fitness-solu_len*0.3
    
    def modify_sentence(self, solution):
        edited_sentence = list(self.tokens)
        selected_tokens = []

        for i, gene in enumerate(solution):
            if gene > 0 and len(selected_tokens) < self.max_edits:  # Enforce max edits

                half_token_len=len(edited_sentence[i])//2
                if half_token_len<=1:
                    continue

                selected_tokens.append(i)

                # # Treat operation as part of the solution
                # operation = solution[len(self.tokens) + i]  # Operation encoded in the extended solution
                # operation=gene
                operation = 2 #random.choice([1, 2, 3])

                tmp_token=edited_sentence[i]

                if operation == 1:  # Delete
                    edited_sentence[i] = tmp_token[:half_token_len] + tmp_token[half_token_len+1:]
                elif operation == 2:  # Replace
                    tmp_char=tmp_token[half_token_len]
                    edited_sentence[i] = tmp_token[:half_token_len] +find_homo(tmp_char)+ tmp_token[half_token_len+1:] 
                elif operation == 3:  # Insert
                    edited_sentence[i] = tmp_token[:half_token_len] + self.special_char+ tmp_token[half_token_len:]

        # Reconstruct sentence
        modified_sentence = " ".join(edited_sentence)
        edit_distance=len(selected_tokens)
        return modified_sentence, edit_distance, np.abs(edit_distance-self.max_edits)

    def fitness_function(self, ga_instance, solution, solution_idx):
        
        modified_sentence, _ , solu_len= self.modify_sentence(solution)

        # Evaluate fitness using the helper function
        return self.evaluate_fitness(modified_sentence, self.target_class, solu_len)

    def get_adv(self, sentence, target_class, num_generations=30, num_parents_mating=10, population_size=100, max_edit_rate=0.1, mutation_percent_genes=30):
        
        self.tokens = sentence.split()
        self.target_class = target_class
        self.max_edits = max(1, int(len(self.tokens) * max_edit_rate))  # Set max_edits to 30% of the token count
        n = len(self.tokens)

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
        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        best_sentence, edit_distance, solu_len = self.modify_sentence(best_solution)
        return best_sentence, edit_distance, best_solution_fitness

    def on_generation(self, ga_instance):
        # self.log_info(f"Generation: {ga_instance.generations_completed}")
        (best_solution, best_fitness, _)=ga_instance.best_solution()
        self.log_info(f"Generation: {ga_instance.generations_completed}, Best Fitness: {best_fitness}")
        if self.wm_detector is not None:
            best_sentence,_, solu_len = self.modify_sentence(best_solution)
            wm_rlt=self.wm_detector(best_sentence)
            self.log_info(f"evl Detect: {wm_rlt}")
            self.log_info(f"tmp best_sentence: {best_sentence}")

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
