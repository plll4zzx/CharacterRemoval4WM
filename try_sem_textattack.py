

import textattack
import transformers
from read_data import c4
import textattack.attack_sems


def main(
    dataset_name = '../../dataset/c4/realnewslike',
    attack_name = 'TextFoolerJin2019', 
    victim_name = 'sentence-transformers/all-mpnet-base-v2',
    file_num=10,
    file_data_num=2,
    target_cos=0.7, 
    edit_distance=10,
    text_len=50
):

    dataset_name='../../dataset/c4/realnewslike'
    file_num=int(file_num)

    c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num)
    c4_dataset.load_data(text_len)
    
    dataset = textattack.datasets.Dataset(c4_dataset.data)
    
    model = transformers.AutoModel.from_pretrained(victim_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(victim_name)
    model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(model, tokenizer)
    attack = getattr(textattack.attack_recipes, attack_name).build(model_wrapper, target_cos=target_cos, edit_distance=edit_distance)

    attack_args = textattack.AttackArgs(
        num_examples=int(file_num*file_data_num),
        attack_name=attack_name,
        dataset_name=dataset_name.replace('/','_'),
        victim_name=victim_name,
        query_budget=1000,
        disable_stdout=False,
        parallel=False
    )
    
    attacker = textattack.Attacker(attack, dataset, attack_args)
    # attacker.attack.attack(c4_dataset.data[0][0], c4_dataset.data[0][1])
    attack_results = attacker.attack_dataset()
    print()


if __name__ == '__main__':
    for attack_name in ['DeepWordBugGao2018']:
        #'CLARE2020','A2TYoo2021', 'PWWSRen2019', 'TextFoolerJin2019', 'BAEGarg2019', 'TextBuggerLi2018', 'DeepWordBugGao2018', 'BERTAttackLi2020'
        print(attack_name)
        main(
            dataset_name = '../../dataset/c4/realnewslike',
            attack_name = attack_name, 
            victim_name = 'sentence-transformers/all-mpnet-base-v2',
            file_num=10,
            file_data_num=2,
            target_cos=0.7, 
            edit_distance=10,
            text_len=50
        )
        print(attack_name)