from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B"  # Replace with actual available model name if required
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


model.save_pretrained("./models/llama31")
tokenizer.save_pretrained("./models/llama31")
