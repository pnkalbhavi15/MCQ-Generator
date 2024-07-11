#mcq generator using fine-tuned model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import random
import gradio as gr
import spacy

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

# def generate_distractors(correct_answer, context, num_distractors=3):
#     try:
#         prompt = f"Generate {num_distractors} distractors for the answer '{correct_answer}' in the context:\n{context}\nDistractors:"
#         inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True)
#         outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         distractors = generated_text.split("Distractors:")[1].strip().split("\n")
#         return [distractor.strip() for distractor in distractors if distractor.strip()]
    
#     except Exception as e:
#         print(f"Error generating distractors: {e}")
#         return []

def get_similar_words(word, max_results=10):
    try:
        response = requests.get(f"https://api.datamuse.com/words?ml={word}&max={max_results}")
        if response.status_code == 200:
            words = [item['word'] for item in response.json()]
            return words
    except Exception as e:
        print(f"Error fetching similar words: {e}")
    return []

def generate_question_and_mcq(content, num_questions=1):
    try:
        inputs = tokenizer.encode(content, return_tensors="pt", max_length=512, truncation=True)
        prompt = tokenizer.decode(inputs[0], skip_special_tokens=True)
        
        questions = set()
        while len(questions) < num_questions:
            doc = nlp(content)
            # entities = tokenizer.decode(inputs[0], skip_special_tokens=True).split()
            entities = [ent.text for ent in doc.ents]
            entity = random.choice(entities)
            
            doc = nlp(content)
            sentences = [sent.text for sent in doc.sents]
            context_sentence = next((sent for sent in sentences if entity in sent), sentences[0])
            question = context_sentence.replace(entity, "______")
            
            distractors = get_similar_words(entity)
            while len(distractors) < 3:
                random_word = tokenizer.decode(model.generate(tokenizer.encode("generate a random word", return_tensors="pt", padding=True), max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
                if random_word.lower() != entity.lower() and random_word not in distractors:
                    distractors.append(random_word)
                    
            distractors = distractors[:3]
            options = distractors + [entity]
            random.shuffle(options)
            
            generated_question = f"Question: {question}\n\nOptions:\n"
            for i, option in enumerate(options, 1):
                generated_question += f"{i}. {option}\n"
                
            generated_question += f"\nAnswer: {entity}"
            questions.add(generated_question)
            entities.remove(entity)
        
        return "\n\n".join(questions)
    
    except Exception as e:
        print(f"Error generating MCQs: {e}")
        return f"Error: {str(e)}"

def generate_mcqs(content, num_questions):
    return generate_question_and_mcq(content, num_questions=num_questions)

iface = gr.Interface(
    fn=generate_mcqs,
    inputs=[
        gr.Textbox(label="Enter the content to generate MCQs", placeholder="Type here..."),
        gr.Number(label="Number of MCQs to generate", value=1, minimum=1, maximum=10, step=1)
    ],
    outputs=gr.Textbox(label="Generated MCQs"),
    title="MCQ Generator",
    description="Generate multiple-choice questions (MCQs) based on input text content."
)

iface.launch()
