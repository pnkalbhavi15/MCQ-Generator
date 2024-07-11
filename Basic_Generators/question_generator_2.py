#generates multiple questions sentence-wise from given content
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import spacy

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

def generate_question(sentence, entity, question_number):
    if entity.lower() in sentence.lower():
        question = sentence.lower().replace(entity.lower(), "______")
        question = question.capitalize()
    else:
        question = f"Generate a question from the given sentence: '{sentence}' with answer as '{entity}'?"

    input_ids = tokenizer.encode(question, return_tensors="pt")
    max_length = min(512, len(input_ids[0]) + 50)
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return generated_question

def generate_questions(context, num_questions=1):
    try:
        doc = nlp(context)
        sentences = list(doc.sents)
        questions = []
        entities_used = set()
        question_number = 1

        for sentence in sentences:
            entities = list(set([ent.text for ent in sentence.ents if ent.label_ not in ['DATE', 'CARDINAL']]))
            if not entities:
                continue

            for entity in entities:
                if len(questions) >= num_questions:
                    break

                if entity not in entities_used:
                    question = generate_question(sentence.text, entity, question_number)
                    questions.append(f"Question {question_number}: {question}")
                    entities_used.add(entity)
                    question_number += 1

            if len(questions) >= num_questions:
                break

        if len(questions) < num_questions:
            questions.append(f"Only {len(questions)} questions could be generated from the provided context.")
        return "\n".join(questions)

    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=generate_questions,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter a context here..."),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Questions")
    ],
    outputs=gr.Textbox(label="Generated Questions", lines=15),
    title="Question Generator",
    description="Enter a context to generate questions based on the input content. Each question will be labeled."
)
interface.launch()
