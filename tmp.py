# from transformers import (
#     AutoTokenizer, BayesianDetectorModel, SynthIDTextWatermarkLogitsProcessor, SynthIDTextWatermarkDetector
# )

# # Load the detector. See examples/research_projects/synthid_text for training a detector.
# detector_model = BayesianDetectorModel.from_pretrained("joaogante/dummy_synthid_detector")
# logits_processor = SynthIDTextWatermarkLogitsProcessor(
#     **detector_model.config.watermarking_config, device="cpu"
# )
# tokenizer = AutoTokenizer.from_pretrained(detector_model.config.model_name)
# detector = SynthIDTextWatermarkDetector(detector_model, logits_processor, tokenizer)

# # Test whether a certain string is watermarked
# test_input = tokenizer(["This is a test input"], return_tensors="pt")
# is_watermarked = detector(test_input.input_ids)

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
x = np.random.randn(1000) * 10 + 50
y = np.random.randn(1000) * 10 + 50

import seaborn as sns

plt.figure(figsize=(8, 6))

# 绘制 KDE 密度背景
sns.kdeplot(x=x, y=y, cmap="Blues", fill=True)

# 叠加散点图
plt.scatter(x, y, color="black", alpha=0.3, s=10)

# 标注
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot with KDE Density")
# plt.show()

from evaluate import load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 Perplexity 评估工具
perplexity = load("perplexity", module_type="metric")

# 加载 GPT-2 作为评估模型
model_id = "facebook/opt-1.3b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def calculate_perplexity(text):
    """计算给定文本的 Perplexity"""
    encodings = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss  # 交叉熵损失

    perplexity = torch.exp(loss)  # PPL = e^loss
    return perplexity.item()

# 示例文本
text = "in the region and prosecution of tհose responsible for these shocking acts, vowing that we will pursue justice aɡaіnst tհese terror groups. BⲅEA𝒌INɡ: Over 3,500 child soldiers recruited within tհe northeast of Nigeria since 13th Dec – 3/5. The children are #injured, forcibly dіspⅼacеd & o𝚏ten end up with grave injuries. We urge the Government to urgently ensure that aⅼl child soldiers in northeast regions are protected and their rights are respected. #ChildrenDontFight pіc.twittеr.com/qϳU2E2OJ5G — UոIϲEFNG (@UNICEFNG) September 6, 2019 Children join the conflict in different ways, but have mostly been the vіc𝚝ims of sеxսal a𝚝tɑcks, kidnapping, recruitment, sexual exploitation"
ppl = calculate_perplexity(text)
print(f"Perplexity: {ppl}")