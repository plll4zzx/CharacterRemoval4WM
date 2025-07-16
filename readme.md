##

### Hardware dependencies

A commodity desktop machine with at least 8 CPU cores and 16GB RAM. GPU (e.g., NVIDIA GPU with CUDA support) is strongly recommended for faster execution, especially for batch evaluations and reference detector training.

### Software dependencies

Python 3.9, Other dependencies listed in requirements.txt.

### Benchmarks

We use the C4 dataset as the source of prompts to query the target LLMs and generate watermarked text. The C4 dataset is publicly available at [C4](https://huggingface.co/datasets/allenai/c4), and we recommend downloading it via git for convenience. In this evaluation, we employ OPT-1.3B as the language models to produce watermarked outputs.
Our reference detectors are finetuned from Bert.
For sentence-level attacks, we leverage the [DIPPER](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl) model as a paraphraser. All model weights can be obtained from Hugging Face model repositories


