#generates some-what related 1 question from a sentence
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

def generate_question(sentence):
    try:
        question_prompt = f"Based on the following sentence, generate a relevant question:\n\nSentence: {sentence}\n\nQuestion:"

        input_ids = tokenizer.encode(question_prompt, return_tensors="pt")
        attention_mask = input_ids.clone().detach()
        attention_mask[:] = 1

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        generated_question = post_process_question(generated_question)

        return generated_question

    except Exception as e:
        return f"Error: {str(e)}"

def post_process_question(question):
    if question.startswith("Question:"):
        question = question[len("Question:"):].strip()
    if not question.endswith("?"):
        question += "?"
    return question

interface = gr.Interface(
    fn=generate_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence here..."),
    outputs=gr.Textbox(label="Generated Question"),
    title="Question Generator",
    description="Enter a sentence to generate a question based on the input content."
)
interface.launch()
