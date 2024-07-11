from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import spacy
import random
import requests

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def get_similar_words(word, max_results=10):
    response = requests.get(f"https://api.datamuse.com/words?ml={word}&max={max_results}")
    if response.status_code == 200:
        words = [item['word'] for item in response.json()]
        return words
    return []

def generate_question_and_mcq(sentence):
    try:
        doc = nlp(sentence)
        entities = [ent.text for ent in doc.ents]

        if not entities:
            return "No entities found in the input sentence."
        
        answer = random.choice(entities)
        question = sentence.replace(answer, "______")

        distractors = get_similar_words(answer, max_results=10)
        
        while len(distractors) < 3:
            random_word = tokenizer.decode(model.generate(tokenizer.encode("generate a random word", return_tensors="pt"), max_new_tokens=1, num_return_sequences=1)[0], skip_special_tokens=True)
            if random_word.lower() != answer.lower() and random_word not in distractors:
                distractors.append(random_word)
        
        distractors = distractors[:3]
        
        options = distractors + [answer]
        random.shuffle(options)

        generated_question = f"Question: {question}\n\nOptions:\n"
        for i, option in enumerate(options, 1):
            generated_question += f"{i}. {option}\n"
        
        generated_question += f"\nAnswer: {answer}"
        
        return generated_question
    
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=generate_question_and_mcq,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence here..."),
    outputs=gr.Textbox(label="Generated MCQ"),
    title="MCQ Generator",
    description="Enter a sentence to generate a multiple-choice question based on the input content."
)
interface.launch()
