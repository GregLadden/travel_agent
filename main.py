import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

# Initialize the pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text with explicit settings to avoid warnings
prompt = "Suggest a hidden gem in Europe for a family-friendly vacation, with details on activities for kids and dining options."
result = pipe(
    prompt,
    max_length=1000,
    num_return_sequences=1,
    pad_token_id=pipe.tokenizer.eos_token_id,
    truncation=True,
    temperature=0.7,
    top_p=0.9,
    no_repeat_ngram_size=3  # Prevents repeating trigrams
)

print(result[0]["generated_text"])
