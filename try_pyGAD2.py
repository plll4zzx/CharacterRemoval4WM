import random
import numpy as np
from pygad import GA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class GA_Attack:
    def __init__(
        self, 
        victim_model = 'bert-base-uncased', 
        victim_tokenizer = 'bert-base-uncased',
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(victim_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(victim_model)
        self.special_char = ' '

    def evaluate_fitness(self, modified_sentence, target_class):
        # Tokenize the modified sentence
        inputs = self.tokenizer(modified_sentence, return_tensors="pt", padding=True, truncation=True)

        # Evaluate fitness using the fine-tuned classification model
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)

        # Define a fitness value based on the target misclassification
        fitness = predictions[0][target_class].item()

        return fitness

    def fitness_function(self, ga_instance, solution, solution_idx):
        # Decode solution into token indices and operations
        edited_sentence = list(self.tokens)
        selected_tokens = []

        for i, gene in enumerate(solution):
            if gene == 1 and len(selected_tokens) < self.max_edits:  # Enforce max edits
                selected_tokens.append(i)
                operation = random.choice(["delete", "replace", "insert"])
                token_index = i
                if operation == "delete":
                    edited_sentence[token_index] = edited_sentence[token_index][:-1]  # Remove last character
                elif operation == "replace":
                    edited_sentence[token_index] = self.special_char  # Replace entire token with special character
                elif operation == "insert":
                    edited_sentence[token_index] = edited_sentence[token_index] + self.special_char  # Insert special char

        # Reconstruct sentence
        modified_sentence = " ".join(edited_sentence)

        # Evaluate fitness using the helper function
        return self.evaluate_fitness(modified_sentence, self.target_class)

    def get_adv(self, sentence, target_class, num_generations=30, num_parents_mating=10, population_size=100):
        self.tokens = sentence.split()
        self.target_class = target_class
        self.max_edits = max(1, int(len(self.tokens) * 0.3))  # Set max_edits to 30% of the token count
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
            init_range_high=2,
            mutation_percent_genes=30,
            on_generation=self.on_generation,
        )

        # Run the algorithm
        ga_instance.run()

        # Output the best solution
        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        return best_solution, best_solution_fitness

    def on_generation(self, ga_instance):
        print(f"Generation: {ga_instance.generations_completed}")
        print(f"Best Fitness: {ga_instance.best_solution()[1]}")

# Example usage
if __name__ == "__main__":
    ga_attack = GA_Attack(
        victim_model = 'saved_model/RefDetector_KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2024-12-23',
        victim_tokenizer = 'bert-base-uncased'
    )
    sentence = "attempt at promotion to english football ’ s second tier. \" having been relegated from the championship ( south preston had struggled at the wrong end ) to this campaign and fighting back through promotion gives us great confidence, \" said lewer – a league one winner with barnsley in 1997, twice a northern irish premier league winner with hamilton and with northampton town, as well as a top - flight player. sign up to our daily newsletter the i newsletter cut through the noise sign up thanks for signing up! sorry, there seem to be some issues. please try again later. submitting... blackpool manager alan lewier gives instructions to his team from the technical area during blackpool v rotherham united match at bloomfield road, blackpool, saturday may 29, 2021. ( photo by tony johnson ). the seasiders kicked off the new league one season with a 1 - 0 success at rotherham, but blackpool's 3 - 3 draw with oldham, with"

    target_class = 1  # Target class index to maximize
    ga_attack.evaluate_fitness(sentence, target_class)
    best_solution, best_fitness = ga_attack.get_adv(sentence, target_class)
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
