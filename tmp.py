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

# 生成示例数据
np.random.seed(42)
x = np.linspace(-100, 100, 50)  # 真实值，可以为负数
y = x + np.random.normal(0, 20, size=x.shape)  # 预测值（带有噪声）

def plot_scatter(true_values, predicted_values, fig_path=None):
    if len(true_values)>1000:
        sample_indices = np.random.choice(len(true_values), size=1000, replace=False) 
        true_values=true_values[sample_indices]
        predicted_values=predicted_values[sample_indices]
    min_value=min(true_values.min(), predicted_values.min())
    max_value=max(true_values.max(), predicted_values.max())
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predicted_values, color='blue', alpha=0.6, label="Predictions")
    tmp_max=max(abs(max_value), abs(min_value))
    plt.plot([-tmp_max, tmp_max], [-tmp_max, tmp_max], color='red', linestyle='--', label="Ideal Line (y=x)")  # 参考对角线

    # 图表美化
    plt.xlabel("Original Watermark Score")
    plt.ylabel("Reference Watermark Score")
    # plt.fill_betweenx([-tmp_max, tmp_max], 0, -tmp_max, color='lightgrey', alpha=0.5)
    plt.fill_betweenx([-tmp_max, tmp_max], 0, tmp_max, where=[False, False], color='lightgrey', alpha=0.5)
    x_fill = np.linspace(-tmp_max, 0, 100)  # 取 x 轴左半部分
    plt.fill_between(x_fill, 0, tmp_max, color='lightgray', alpha=0.5)
    x_fill = np.linspace(0, tmp_max, 100)  # 取 x 轴右半部分
    plt.fill_between(x_fill, -tmp_max, 0, color='lightgray', alpha=0.5)

    # plt.title("Scatter Plot of True vs Predicted Values")
    # plt.legend()
    plt.grid(True)

    
    # 设置坐标轴加粗
    ax = plt.gca()
    # ax.spines['top'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    
    # 将原点放在图像正中央
    ax.set_xlim([-tmp_max, tmp_max])
    ax.set_ylim([-tmp_max, tmp_max])
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    plt.show()
plot_scatter(x, y)