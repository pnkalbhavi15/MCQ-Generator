from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import random
import gradio as gr
import pdfplumber
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")

tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

tokenizer.pad_token = tokenizer.eos_token

def get_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

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
        inputs = tokenizer.encode_plus(content, return_tensors="pt", max_length=512, truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        questions = set()
        while len(questions) < num_questions:
            doc = nlp(content)
            entities = [ent.text for ent in doc.ents]
            if not entities:
                break
            entity = random.choice(entities)
            
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

def generate_mcqs_from_pdf(file_path, num_questions):
    try:
        pdf_text = get_text_from_pdf(file_path)
        if pdf_text:
            return generate_question_and_mcq(pdf_text, num_questions=num_questions)
        else:
            return "Error: Unable to extract text from PDF."
    except Exception as e:
        print(f"Error generating MCQs from PDF: {e}")
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=generate_mcqs_from_pdf,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"),
        gr.Number(label="Number of MCQs to generate", value=1, minimum=1, maximum=10, step=1)
    ],
    outputs=gr.Textbox(label="Generated MCQs"),
    title="MCQ Generator from PDF",
    description="Generate multiple-choice questions (MCQs) based on text extracted from a PDF file using pdfplumber and spaCy."
)
iface.launch()
