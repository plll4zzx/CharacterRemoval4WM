## Character-Level Perturbations in Disrupt LLM Watermark

We provide a scaled-down version of the experiments to validate the two primary claims presented in the paper: (1) character-level perturbations achieve superior watermark removal effectiveness compared to token-level perturbations under the same editing rate, and (2) genetic algorithm-based optimization can leverage the guidance of a reference detector to further improve removal attack performance.

### Hardware dependencies

A commodity desktop machine with at least 8 CPU cores and 16GB RAM. GPU (e.g., NVIDIA GPU with CUDA support) is strongly recommended for faster execution, especially for batch evaluations and reference detector training.

### Software dependencies

Python 3.9, Other dependencies listed in requirements.txt.

```
pip install -r requirements.txt
```

### Dataset and Model

We use the C4 dataset as the source of prompts to query the target LLMs and generate watermarked text. The C4 dataset is publicly available at [C4](https://huggingface.co/datasets/allenai/c4).

We recommend downloading it via git for convenience. The commands are:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include realnewslike/*
```

Our reference detectors are finetuned from [Bert](https://huggingface.co/google-bert/bert-base-uncased).
For sentence-level attacks, we leverage the [DIPPER](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl) model as a paraphraser. All model weights can be obtained from Hugging Face model repositories

### Experiment Workflow

* Baseline Removal Evaluation: Execute the scripts for random attacks to validate that character-level perturbations outperform token-level perturbations in removing watermark. This step corresponds to the results reported in Tables II of the paper.
* Guided Removal Evaluation: Train reference detectors on dataset of the watermarked data. Then, run the guided removal attacks, including Best-of-N and Genetic Algorithm (GA) -based optimization, to demonstrate their superior performance.

### Baseline Removal Evaluation

**Preparation**
To run this experiment, the C4 dataset is needed. We recommend storing it in the path "../../dataset/c4/realnewslike", where ".." refers to the parent directory relative to the current working directory. Watermarked text is generated using the OPT-1.3B model. The model weights do not require a separate download, as they are automatically retrieved during script execution.
This process can be performed with the script collect_wm_text.py, which requires specifying the watermark name, dataset path, model name, the number of text, and the GPU device. For example:

```
python collect_wm_text.py --wm_name "KGW" \
--dataset_name "../../dataset/c4/realnewslike" \
--model_name "facebook/opt-1.3b" --device 0\
--file_num 50 --file_data_num 1000 
```

This command generates 5000 (file_num * file_data_num) watermarked samples using the "facebook/opt-1.3b" model with the KGW watermark on GPU device 0. To produce watermarked text with other watermarking schemes, simply modify the --wm_name parameter. We recommend generating at least 5000 examples to support reference detector training

**Execution**

```
python test_rand_sh.py \
--llm_name "facebook/opt-1.3b" \
--wm_name_list "['KGW']" \
--atk_style_list "['token','char']" \
--max_edit_rate_list "[0.1, 0.5]" \
--do_flag "True" --atk_times_list "[1]" \
--max_token_num_list "[100]"
```

In this example, the editing rates are set to 0.1 and 0.5, and the length of the watermarked text is fixed to 100 tokens, consistent with the settings reported in the paper. As this is a random strategy, the atk_times_list parameter is set to 1. To evaluate additional watermarking schemes, simply add their names to the wm_name_list.

The results of this script are saved as log files in "attack_log/Rand" and as JSON files in "saved_attk_data". Alternatively, setting the "do_flag" parameter to "False" prints the results directly to the terminal. The output demonstrates that, for the same editing rate, character-level perturbations consistently achieve higher watermark score dropping rate (WDR) and attack success rates (ASR) compared to token-level perturbations.

### Guided Removal Evaluation

**Preparation**
Train the reference detector:

```
python train_ref_detector.py  --device 0\
--wm_name "KGW" --num_epochs 15\
--rand_char_rate 0.15 --rand_times 9\
--llm_name "facebook/opt-1.3b" --ths 4
```

In this example, the watermarking scheme is set to KGW. Each sample is augmented 9 times, with an editing rate of 0.15 ("rand_char_rate"). The "ths" is detection threshold used to evaluate the performance of the reference detector. The number of training epochs is set to 15. Due to both data augmentation and model training are computationally intensive and time-consuming, preprocessed datasets and pre-trained reference detector are provided in the "saved_data" and "saved_model" to facilitate artifact evaluation.

**Execution**
Best-of-N:

```
python test_rand_sh.py \
--llm_name "facebook/opt-1.3b" \ 
--wm_name_list "['KGW']" \
--atk_style_list "['token','char']" \
--data_aug_list "[9]" --max_token_num_list "[100]"\
--max_edit_rate_list "[0.1]" \
--do_flag "True" --atk_times_list "[10]" \
```

the atk_times_list parameter specifies the value of $N$ in the Best-of-$N$ strategy, determining how many perturbation candidates are sampled for each input. The data_aug_list is used to choose reference detector. Due to we set rand_times to 9 when training the reference detector, data_aug_list is also set to 9 here to maintain consistency.

Sand:

```
python test_rand_sh.py \
    --llm_name "facebook/opt-1.3b" \ 
    --wm_name_list "['KGW']" \
    --atk_style_list "['sand_token','sand_char']" \
    --data_aug_list "[9]" --do_flag "True" \
    --max_edit_rate_list "[0.1]" \
    --atk_times_list "[1]" \
    --max_token_num_list "[100]"
```



GA:

```
python test_ga_sh.py \
--llm_name "facebook/opt-1.3b" \
--wm_name "KGW" --atk_style "char" \
--data_aug 9 --do_flag "True" \
--num_generations 15 --max_edit_rate 0.13 \
--max_token_num_list "[100]"
```

the results of these three methods are saved by default as log files in the "attack_log/Rand" and "attack_log/GA" directories and as json files in the "saved_attk_data" folder. Setting the "do_flag" parameter to "False" will print the results directly to the terminal.
