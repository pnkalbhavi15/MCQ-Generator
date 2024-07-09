from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
app = FastAPI()
# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")
# Define a Pydantic model for the input text
class InputText(BaseModel):
    text: str
# Function to generate questions using SpaCy for NER
def generate_questions_logic(input_text: str):
    try:
        # Process text with SpaCy
        doc = nlp(input_text)
        
        # Example: Generate questions based on named entities
        questions = []
        for ent in doc.ents:
            questions.append(f"What is {ent.text}?")
        
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# API endpoint for generating questions
@app.post("/generate_questions/")
def generate_questions(input_text: InputText):
    try:
        # Call your question generation logic
        questions = generate_questions_logic(input_text.text)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
