from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")

def load_new_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

new_dataset = load_new_dataset("../SQuAD_Datasets/train-v1.1-2000_to_3000.json", tokenizer)

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
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=new_dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
