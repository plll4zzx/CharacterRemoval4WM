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

# import matplotlib.pyplot as plt
# import numpy as np

# import numpy as np
# import matplotlib.pyplot as plt

# # 生成示例数据
# np.random.seed(42)
# x = np.random.randn(1000) * 10 + 50
# y = np.random.randn(1000) * 10 + 50

# import seaborn as sns

# plt.figure(figsize=(8, 6))

# # 绘制 KDE 密度背景
# sns.kdeplot(x=x, y=y, cmap="Blues", fill=True)

# # 叠加散点图
# plt.scatter(x, y, color="black", alpha=0.3, s=10)

# # 标注
# plt.xlabel("X values")
# plt.ylabel("Y values")
# plt.title("Scatter Plot with KDE Density")
# plt.show()

import os

# os.environ["HF_HOME"] = "/mnt/codedisk/huggingface/huggingface"
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/codedisk/huggingface/huggingface/hub"
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")