import fitz  # PyMuPDF
import re
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
def extract_text_from_pdf(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)  # Page numbers are 0-based in PyMuPDF
    text = page.get_text()
    return text
def clean_text(text):
    # Remove special characters and multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
def generate_questions(context, pipeline, num_questions=5):
    # Prompt the model to generate questions
    input_text = f"Generate questions based on the following context:\n\n{context}\n\nQ:"
    results = pipeline(input_text, max_length=150, num_return_sequences=num_questions, num_beams=5)
    
    questions = [result['generated_text'] for result in results]
    return questions
def main(pdf_path, page_number):
    print(f"Extracting text from page {page_number} of {pdf_path}...")
    pdf_text = extract_text_from_pdf(pdf_path, page_number)
    cleaned_text = clean_text(pdf_text)
    print("Generating questions...")
    questions = generate_questions(cleaned_text, llama_pipeline)
    print("\nGenerated Questions:")
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
if __name__ == "__main__":
    pdf_path = r"C:\Users\ssspr\Desktop\MCQ\MCQ-Generator\MESnotes.pdf"
    page_number = 5
    print("Loading model...")
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')
    model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b')
    llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    main(pdf_path, page_number)