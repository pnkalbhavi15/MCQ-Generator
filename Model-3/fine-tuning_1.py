from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

def load_dataset(file_path, tokenizer, block_size=128):
    print(f"Loading dataset from {file_path}...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

try:
    train_dataset = load_dataset("../SQuAD_Datasets/train-v1.1-0_to_100.json", tokenizer)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',            
    logging_steps=200                
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print("Starting training...")
try:
    trainer.train()
    print("Training completed.")
except Exception as e:
    print(f"Error during training: {e}")
    raise

print("Saving the fine-tuned model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Model saved.")
