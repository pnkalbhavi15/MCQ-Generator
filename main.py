import fitz  # PyMuPDF for PDF text extraction
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Use the provided pdf_path parameter
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text
def rag_pipeline(input_text):
    # Initialize RAG components
    tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
    retriever = RagRetriever.from_pretrained('facebook/rag-token-nq')
    model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq')
    # Retrieve relevant passages based on input_text
    retrieved_passages = retriever(input_text)
    # Augment input_text with retrieved passages
    augmented_input = f"{input_text} {retrieved_passages['text']}"
    # Generate questions based on augmented input
    inputs = tokenizer(augmented_input, return_tensors="pt")
    generated_questions = model.generate(inputs['input_ids'])
    # Decode and print generated questions
    print("\nGenerated Questions:")
    for i, generated_question in enumerate(generated_questions, 1):
        question = tokenizer.decode(generated_question, skip_special_tokens=True)
        print(f"Q{i}: {question}")
# Main function
if __name__ == "__main__":
    pdf_path = r"C:\Users\ssspr\Desktop\MCQ\MCQ-Generator\MESnotes.pdf"  # Replace with your PDF file path
    print(f"Extracting text from {pdf_path}...")
    input_text = extract_text_from_pdf(pdf_path)
    # Run RAG pipeline on extracted text
    print("Running RAG pipeline...")
    rag_pipeline(input_text)