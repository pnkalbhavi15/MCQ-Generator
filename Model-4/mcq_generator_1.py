# To be improved
# Currently generates questions using langchain and fine-tuned gpt2
# options using api, if not llm
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import requests
import gradio as gr

model_name = "../Model-2/fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)

text_generation_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)

prompt_template = PromptTemplate(
    template="Generate a question and its answer based on the following content:\n\n{text}\n\nQuestion:",
    input_variables=["text"]
)

llm_chain = LLMChain(llm=text_generation_pipeline, prompt=prompt_template)

def extract_question_answer(generated_text):
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    question = None
    answer = None
    
    if "Question:" and "Answer:" in generated_text:
        flag = True
    flag = False
    for line in lines:
        if line.startswith('Question:'):
            question = line[len('Question:'):].strip()
        if line.startswith('"question":'):
            question = line[len('"question":'):].strip()
        if flag and question and line.startswith('Answer:'):
            answer = line[len('Answer:'):].strip()
            break
        if question and line.startswith('"text":'):
            answer = line.split('"text": ')[-1].strip().strip('"')
            break
    
    print("Question:", question)
    print("Answer:", answer)
    
    return question, answer

def generate_mcq(text, num_questions=5):
    mcqs = []
    while len(mcqs) < num_questions:
        qa_pairs = llm_chain({"text": text})

        generated_text = qa_pairs['text'].strip()

        question, answer = extract_question_answer(generated_text)

        if question and answer:
            options = generate_options(answer, text)
            options.insert(0, answer)  
            random.shuffle(options)   
            if options:
                mcq = {
                    "question": question,
                    "options": options,
                    "answer": answer
                }
                mcqs.append(mcq)
    return mcqs

def generate_options(correct_answer, text, num_options=3):
    distractors = get_similar_words(correct_answer)
    while len(distractors) < 3:
        # random_word = tokenizer.decode(model.generate(tokenizer.encode("generate a random word", return_tensors="pt", padding=True), max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
        # if random_word.lower() != correct_answer.lower() and random_word not in distractors:
        #     distractors.append(random_word.capitalize())
        distractors.append(generate_options_using_model(correct_answer, text))
    return distractors[:3]

def get_similar_words(word, max_results=10):
    try:
        response = requests.get(f"https://api.datamuse.com/words?ml={word}&max={max_results}")
        if response.status_code == 200:
            words = [item['word'] for item in response.json()]
            return words
    except Exception as e:
        print(f"Error fetching similar words: {e}")
    return []

def generate_options_using_model(correct_answer, text, num_options=3):
    options = []
    for _ in range(num_options):
        prompt = f"In the context - {text}. Generate contextually related alternative for - {correct_answer} \n\nDistractor:"
        response = text_generation_pipeline(prompt)

        distractor = extract_distractor(response)
        options.append(distractor)
    return options

def extract_distractor(response):
    if isinstance(response, str):
        lines = response.split("\n")
        for line in lines:
            if line.strip().startswith("Distractor:"):
                return line.strip().lstrip("Distractor:").strip()

def generate_mcq_interface(text, num_questions):
    mcqs = generate_mcq(text, num_questions)

    result = ""
    for i, mcq in enumerate(mcqs, 1):
        result += f"Question {i}: {mcq['question']}\n"
        for idx, option in enumerate(mcq['options'], 1):
            result += f"Option {idx}: {option}\n"
        result += "\n " + "Answer: " + mcq['answer'] + "\n"
        result += "\n"
    return result

iface = gr.Interface(
    fn=generate_mcq_interface,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter the content here..."),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of questions")
    ],
    outputs="text",
    title="MCQ Generator using LangChain",
    description="Generate multiple choice questions from content using a fine-tuned GPT-2 model and LangChain."
)
iface.launch()
