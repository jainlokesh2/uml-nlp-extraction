from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_model")

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate a sentence
def generate_sentence(keywords):
    inputs = tokenizer(keywords, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_length=50, num_beams=5, temperature=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
print(generate_sentence("Tester, submit, bug, report"))

