# LLM-Fine-Tuning-Demo: Adapting GPT-2 with Hugging Face Transformers

Fine-tuning a Large Language Model (LLM) involves adapting a pretrained model to a specific task or domain by training it further on a smaller, task-specific dataset. This process allows the model to learn task-specific patterns and improve its performance on that task.

In this tutorial, we’ll walk you through the steps to fine-tune an LLM using the Hugging Face transformers library, which provides easy-to-use tools for working with models like GPT, BERT, and others. we’ll also provide a code demo for fine-tuning GPT-2 (a smaller version of GPT-3) on a custom text dataset.

More specifically, in this tutorial, we will fine-tune GPT-2 on the Wikitext dataset (a collection of Wikipedia articles) to generate coherent, context-aware text. You’ll learn how to adapt a pretrained LLM to produce Wikipedia-style content, useful for tasks like knowledge summarization or content generation.

- **Example Input and Output**
```
Input Prompt: "The history of artificial intelligence"
Generated Output: "The history of artificial intelligence dates back to the 1950s, when researchers began exploring the possibility of creating machines that could mimic human intelligence. Early milestones include the development of the first AI programs and the establishment of key theoretical frameworks."
```

## Steps to Fine-Tune an LLM

### Choose a Pretrained Model
- Select a model architecture (e.g., **GPT-2**, **BERT**) that suits your task (e.g., text generation, classification).
- Hugging Face provides a wide range of pretrained models that you can fine-tune for your tasks. Here are some popular models for different tasks:

  - **Text Generation Models**
    - **GPT-Neo** (by EleutherAI): An open-source alternative to GPT-3.

    - **GPT-J:** A 6B parameter model, larger than GPT-2.

    - **BLOOM:** A multilingual LLM with up to 176B parameters.

    - **T5** (Text-to-Text Transfer Transformer): A versatile model for text generation, summarization, translation, etc.

  - **Text Classification Models**
    - **BERT:** Great for tasks like sentiment analysis, question answering, and more.

    - **RoBERTa:** An optimized version of BERT.

    - **DistilBERT:** A smaller, faster version of BERT.

  - **Multimodal Models**
    - **CLIP:** For tasks combining text and images.

    - **Flamingo:** For advanced multimodal tasks.

  - **Code Generation Models**
    - **Codex** (OpenAI): Specialized for code generation (not open-source, but API-based).

    - **CodeGen** (Salesforce): An open-source alternative for code generation.

### Prepare Your Dataset
- Collect and preprocess a dataset relevant to your task. Examples:
  - **Text generation**: A dataset of stories, articles, or dialogues.
  - **Classification**: A labeled dataset of text and corresponding labels.

  For this tutorial, we’ll use the **Wikitext dataset**, a collection of Wikipedia articles that can be used for text generation tasks. **Here’s a small snippet of what the dataset looks like:**
    
    ```
    = Alan Turing =
    Alan Mathison Turing ( 23 June 1912 – 7 June 1954 ) was an English mathematician , computer scientist , logician , cryptanalyst , philosopher , and theoretical biologist . Turing was highly influential in the development of theoretical computer science , providing a formalisation of the concepts of algorithm and computation with the Turing machine , which can be considered a model of a general-purpose computer . Turing is widely considered to be the father of theoretical computer science and artificial intelligence .
    
    = Early Life and Education =
    Turing was born in Maida Vale , London , while his father , Julius Mathison Turing , was on leave from his position with the Indian Civil Service ( ICS ) at Chatrapur , then in the Madras Presidency and presently in Odisha state , India . Turing's father was the son of a clergyman , the Rev. John Robert Turing , from a Scottish family of merchants that had been based in the Netherlands and included a baronet . Turing's mother , Ethel Sara Turing ( née Stoney ) , was the daughter of Edward Waller Stoney , chief engineer of the Madras Railways .
    ```

- **Key Features of the Dataset**
    - **Structure:**

      - Articles are separated by blank lines.
      
      - Section headers are enclosed in = = (e.g., = Alan Turing =).
      
      - Text is clean and well-formatted, making it ideal for training language models.

    - **Content:**

      - The dataset covers a wide range of topics, from biographies to scientific concepts.

      - Sentences are grammatically correct and factually dense, typical of Wikipedia articles.

  - **Use Case:**
    - Fine-tuning on this dataset allows the model to generate Wikipedia-style text, which is useful for tasks like summarization, question answering, or content generation.

- Here’s **how you can load and inspect the dataset** using the Hugging Face datasets library:
  ```python
  from datasets import load_dataset
  
  # Load the Wikitext dataset
  dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
  
  # Print the first 5 examples**
  for i in range(5):
      print(dataset[i]["text"])
      print("-" * 80)
  ```
  
  Running the code will display the first few examples from the dataset, like this:
  
  ```
  = Valkyria Chronicles III = 
  = = 
  Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role-playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . 
  --------------------------------------------------------------------------------
  = = 
  The game is set in the same fictional continent as the previous Valkyria Chronicles games , but follows a new story and characters . The game follows the exploits of Squad 422 , a penal military unit serving the Federation , as they fight against the Imperial forces in the Second Europan War . 
  --------------------------------------------------------------------------------
  = = 
  Development = = 
  Development of Valkyria Chronicles III began in early 2010 , following the release of Valkyria Chronicles II . The game was developed by Sega and Media.Vision , with the original Valkyria Chronicles director , Shuntaro Tanaka , returning as producer . 
  --------------------------------------------------------------------------------
  ```


### Set Up the Training Pipeline
- Use a framework like **Hugging Face** `transformers` to load the pretrained model and tokenizer.
- Define a **data collator** and **training arguments**.

### Fine-Tune the Model
- Train the model on your dataset using a **GPU or TPU** for faster computation.


### Evaluate and Save the Model
- Evaluate the fine-tuned model on a **validation set**.
- Save the model for later use.













### What You Need to Run the Demo

- **Python Environment:**

    Install the required libraries:
    
    ```bash
    pip install transformers datasets accelerate
    ```

- **Hardware:**

    A GPU is highly recommended for fine-tuning. If you don’t have one, you can use cloud services like Google Colab, AWS, or Azure.

- **Dataset:**

    Replace the wikitext dataset with your own dataset. Your dataset should be in a format compatible with Hugging Face’s datasets library (e.g., CSV, JSON, or plain text).
    
    Steps to Adapt the Demo for Your Application

- **Prepare Your Dataset:**

    If your dataset is in a CSV or JSON file, load it like this:

    ```python
    from datasets import load_dataset
    
    # Load a custom dataset
    dataset = load_dataset("csv", data_files={"train": "path/to/your/train.csv", "validation": "path/to/your/val.csv"})
    ```

    Ensure your dataset has a column with text data (e.g., "text").

- **Adjust the Tokenizer:**

    If your dataset contains domain-specific terms, you might need to add special tokens to the tokenizer:

    ```python
    tokenizer.add_tokens(["<SPECIAL_TOKEN>"])
    model.resize_token_embeddings(len(tokenizer))
    ```

- **Modify Training Arguments:**

    Adjust the TrainingArguments based on your hardware and dataset size:

    - per_device_train_batch_size: Reduce this if you run out of GPU memory.
    
    - num_train_epochs: Start with 3–5 epochs and increase if needed.
    
    - max_length: Adjust based on the average length of your text.
    
- **Evaluate the Model:**

    After fine-tuning, evaluate the model on a validation set to ensure it performs well on unseen data.

- **Deploy the Model:**

    Save the fine-tuned model and tokenizer, then integrate them into your application using Hugging Face’s pipeline or custom inference code.



### Full Demo Code (Ready to Run):

```python
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
```


### Generate Text with the Fine-Tuned Model
Okay so now that you have fine-tuned the model, you can generate text using it. Here’s how you can do it:

```python
# Step 9: Generate text with the fine-tuned model
from transformers import pipeline

fine_tuned_model = GPT2LMHeadModel.from_pretrained("./fine-tuned-gpt2")
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-gpt2")

text_generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

prompt = "The future of AI is"
output = text_generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])
```
**Example Output**
After fine-tuning on a dataset of quotes, the model can generate text like this:

```
Input Prompt: "The future of AI is"
Generated Text: "The future of AI is bright, with endless possibilities for innovation and transformation across industries."
```

### How to Use This in Your Application
- **Fine-Tune on Your Dataset:**

    - Replace the wikitext dataset with your own dataset.
    
    - Adjust the max_length and other parameters as needed.

- **Deploy the Model:**

  - Save the fine-tuned model and tokenizer.

  - Load the model in your application using Hugging Face’s pipeline or custom inference code.

- **Optimize for Production:**

  - Use quantization or distillation to reduce the model size and improve inference speed.

  - Deploy the model on a cloud service or edge device.

### Example Use Cases
- **Chatbots:** Fine-tune GPT-2 on conversational data to create a domain-specific chatbot.

- **Content Generation:** Generate product descriptions, blog posts, or social media content.

- **Code Completion:** Fine-tune on code datasets to create a programming assistant.

Hope this helps!