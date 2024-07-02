from PyPDF2 import PdfReader
import pandas as pd
import os

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def save_text_to_csv(text, csv_path):
    df = pd.DataFrame({"text": [text]})
    df.to_csv(csv_path, mode='a', index=False, header=not os.path.isfile(csv_path))

# Provide the path to your PDF file using a raw string literal
pdf_path = r"C:\Users\ssspr\Desktop\MCQ\MCQ-Generator\MESnotes.pdf"
csv_path = "output_text.csv"

# Extract text from PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Save the extracted text to a CSV file
save_text_to_csv(extracted_text, csv_path)

print(f"Text extracted from {pdf_path} and saved to {csv_path}")
