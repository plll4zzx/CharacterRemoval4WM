from transformers import (
    AutoTokenizer, BayesianDetectorModel, SynthIDTextWatermarkLogitsProcessor, SynthIDTextWatermarkDetector
)

# Load the detector. See examples/research_projects/synthid_text for training a detector.
detector_model = BayesianDetectorModel.from_pretrained("joaogante/dummy_synthid_detector")
logits_processor = SynthIDTextWatermarkLogitsProcessor(
    **detector_model.config.watermarking_config, device="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(detector_model.config.model_name)
detector = SynthIDTextWatermarkDetector(detector_model, logits_processor, tokenizer)

# Test whether a certain string is watermarked
test_input = tokenizer(["This is a test input"], return_tensors="pt")
is_watermarked = detector(test_input.input_ids)