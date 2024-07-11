from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import random

tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

prompts = [
    "Based on the following context, generate a factual question:\n\nContext: {context}\n\nQuestion:",
    "Given the following information, what is a possible question?\n\nContext: {context}\n\nQuestion:",
    "Create a question from the following text:\n\nContext: {context}\n\nQuestion:",
    "What question can be asked based on the following passage?\n\nContext: {context}\n\nQuestion:",
]

def generate_questions(context, num_questions=3):
    try:
        questions = set()
        attempts = 0

        while len(questions) < num_questions and attempts < num_questions * 2:
            prompt = random.choice(prompts).format(context=context)
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = input_ids.clone().detach()
            attention_mask[:] = 1

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            generated_question = post_process_question(generated_question)
            if generated_question not in questions and "Beyonc" not in generated_question:
                questions.add(generated_question)
            attempts += 1

        if len(questions) < num_questions:
            return list(questions) + ["Could not generate enough unique questions."]

        return list(questions)

    except Exception as e:
        return [f"Error: {str(e)}"]

def post_process_question(question):
    if "Question:" in question:
        question = question.split("Question:")[-1].strip()
    if not question.endswith("?"):
        question += "?"
    return question

interface = gr.Interface(
    fn=generate_questions,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter context here...", label="Context"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Questions")
    ],
    outputs=gr.Textbox(label="Generated Questions"),
    title="Question Generator",
    description="Enter a context to generate multiple questions based on the input content."
)
interface.launch()
