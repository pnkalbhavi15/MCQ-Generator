#generates 1 question from a simple sentence
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import spacy
import random

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_question(sentence):
    try:
        doc = nlp(sentence)
        entities = [ent.text for ent in doc.ents]

        if not entities:
            return "No entities found in the input sentence."
        
        answer = entities[random.choice(range(len(entities)))]
        if answer.lower() in sentence.lower():
            question = sentence.lower().replace(answer.lower(), "______")
            question = question.capitalize()
        else:
            question = f"Generate a question from the given sentence: {sentence} with answer as {answer}?"

        input_ids = tokenizer.encode(question, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        generated_question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return generated_question
    
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=generate_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence here..."),
    outputs=gr.Textbox(label="Generated Question"),
    title="Question Generator",
    description="Enter a sentence to generate a question based on the input content."
)
interface.launch()