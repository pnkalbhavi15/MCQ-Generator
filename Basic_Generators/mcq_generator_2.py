#generates multiple mcqs from given context
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import spacy
import random
import requests

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token_id = tokenizer.eos_token_id

# def get_distractors_from_gpt2(answer, context, num_distractors=3):
#     prompt = f"Generate {num_distractors} plausible distractors for the answer '{answer}' in the context of the following sentence:\n{context}\nDistractors:"
#     inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True)
#     attention_mask = inputs.ne(tokenizer.pad_token_id).int()
#     outputs = model.generate(inputs, max_length=100, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     distractors = generated_text.split("Distractors:")[1].strip().split("\n")
#     return [distractor.strip() for distractor in distractors if distractor.strip()]

def get_similar_words(word, max_results=10):
    response = requests.get(f"https://api.datamuse.com/words?ml={word}&max={max_results}")
    if response.status_code == 200:
        words = [item['word'] for item in response.json()]
        return words
    return []

def generate_question_and_mcq(content, num_questions=1):
    try:
        doc = nlp(content)
        sentences = [sent.text for sent in doc.sents]
        entities = [ent.text for ent in doc.ents]

        if not entities:
            return "No entities found in the input content."
        
        questions = []
        
        for _ in range(num_questions):
            if not entities:
                break
                
            answer = random.choice(entities)
            context_sentence = next((sent for sent in sentences if answer in sent), sentences[0])
            question = context_sentence.replace(answer, "______")

            # distractors = get_distractors_from_gpt2(answer, context_sentence, num_distractors=3)
            distractors = get_similar_words(answer)

            while len(distractors) < 3:
                random_word = tokenizer.decode(model.generate(tokenizer.encode("generate a random word", return_tensors="pt", padding=True), max_new_tokens=1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
                if random_word.lower() != answer.lower() and random_word not in distractors:
                    distractors.append(random_word)

            distractors = distractors[:3]

            options = distractors + [answer]
            random.shuffle(options)

            generated_question = f"Question: {question}\n\nOptions:\n"
            for i, option in enumerate(options, 1):
                generated_question += f"{i}. {option}\n"

            generated_question += f"\nAnswer: {answer}"
            questions.append(generated_question)
            entities.remove(answer)  

        return "\n\n".join(questions)
    
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=generate_question_and_mcq,
    inputs=[gr.Textbox(lines=10, placeholder="Enter a large content here..."), gr.Number(label="Number of questions")],
    outputs=gr.Textbox(label="Generated MCQs"),
    title="MCQ Generator",
    description="Enter a large content to generate multiple-choice questions based on the input content. Specify the number of questions to generate."
)
interface.launch()
