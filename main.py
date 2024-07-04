from PyPDF2 import PdfReader
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

# Generate questions using LAMA (question-answering pipeline)
def generate_questions(text):
    try:
        nlp = pipeline("question-answering", model="microsoft/LAMA")
        
        # Example queries to indirectly generate questions
        queries = [
            "What is mentioned about...",
            "Explain the concept of...",
            "What are the implications of...",
            "Describe the significance of...",
            "How does ... affect ...",
        ]
        
        questions = []
        for query in queries:
            result = nlp(question=query, context=text)
            answer = result['answer'] if result['score'] > 0 else "N/A"
            questions.append(f"{query} {answer}?")
        
        return questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return None

# Example usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\ssspr\Desktop\MCQ\MCQ-Generator\MESnotes.pdf"
    
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text:
        # Generate questions
        generated_questions = generate_questions(extracted_text)
        
        if generated_questions:
            # Print generated questions
            for i, question in enumerate(generated_questions, start=1):
                print(f"Question {i}: {question}")
        else:
            print("Failed to generate questions.")
    else:
        print("Text extraction failed. Check the PDF file path or format.")
