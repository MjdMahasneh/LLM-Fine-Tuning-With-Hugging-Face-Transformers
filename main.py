# Install required libraries
# pip install transformers datasets accelerate

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# Step 1: Load and prepare the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.select(range(1000))  # Use a subset for quick testing

# Step 2: Load the pretrained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Set up the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # No masked language modeling for GPT-2
)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-gpt2",  # Directory to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=8,   # Batch size per device
    save_steps=500,                  # Save checkpoint every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    evaluation_strategy="steps",     # Evaluate every `eval_steps`
    eval_steps=500,                  # Evaluation frequency
    learning_rate=5e-5,              # Learning rate
    weight_decay=0.01,               # Weight decay
    fp16=True,                       # Use mixed precision (if GPU supports it)
)

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Use the same dataset for evaluation
    data_collator=data_collator,
)

# Step 7: Fine-tune the model
trainer.train()

# Step 8: Save the fine-tuned model
model.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")

# Step 9: Generate text with the fine-tuned model
from transformers import pipeline

fine_tuned_model = GPT2LMHeadModel.from_pretrained("./fine-tuned-gpt2")
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-gpt2")

text_generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

prompt = "Once upon a time"
output = text_generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])



