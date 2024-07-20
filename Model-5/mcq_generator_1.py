# Mcq Generator using Weaviate for vector search, HuggingFace embeddings, LangChain framework, and the Mistral-7B-Instruct-v0.1 model from Hugging Face.
# Based on Prayashi's Collab Notebook
import locale
import weaviate
import random
import gradio as gr
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from secret import WEAVIATE_CLUSTER, WEAVIATE_API_KEY, HUGGING_FACE_API_TOKEN

# Setting encoding
locale.getpreferredencoding = lambda: "UTF-8"

# Initializing Weaviate client
client = weaviate.Client(
    url=WEAVIATE_CLUSTER,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

# Defining the embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Prompting the user to enter their Hugging Face API token
huggingfacehub_api_token = HUGGING_FACE_API_TOKEN

# Initializing HuggingFaceHub model
model = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 1000}
)

def process_pdf(pdf_path):
    try: 
        loader = PyPDFLoader(pdf_path, extract_images=True)
        pages = loader.load()
        
        # Splitting text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        docs = text_splitter.split_documents(pages)

        # print("No of chunks = ", len(docs))
        
        # Creating a vector database
        vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)
        
        return docs, vector_db
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return [], None

def process_text(text):
    try: 
        # Splitting text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_text(text)
        
        # Creating document-like objects for consistency with PDF processing
        docs = [{'page_content': chunk} for chunk in text_chunks]
        
        # Creating a vector database
        vector_db = Weaviate.from_texts(text_chunks, embeddings, client=client)
        
        return docs, vector_db

    except Exception as e:
        print(f"Error processing text: {e}")
        return [], None

# Function to generate MCQs using the model
def generate_mcqs(text_chunk, num_questions):
    prompt = (
        f"Generate {num_questions} multiple-choice questions from the following text:\n\n"
        f"{text_chunk}\n\n"
        "Format the question and options like this:\n"

        "Question: <question text>\n"
        "a) <option 1>\n"
        "b) <option 2>\n"
        "c) <option 3>\n"
        "d) <option 4>\n"
        "Answer: <correct option>\n\n"

        "\nQuestions:"
    )
    response = model(prompt)
    # print("Raw response from the model:", response)
    
    response_text = response.split("Questions:")[-1]
    # print("Response Text:", response_text)
    start_index = response_text.find("1.")
    response_text = response_text[start_index: ]
    questions = response_text.split("\n\n")
    formatted_questions = []
    for question in questions:
        answer_index = question.find("Answer: ")
        question = question[:answer_index + 10]
        if len(question) == answer_index + 9:
            question += ")"
        if len(question) == answer_index + 10  and all(opt in question for opt in ['a)', 'b)', 'c)', 'd)', 'Answer:']):
            if "1." in question:
                question = question.split("2. ")[0]
            elif "2." in question:
                question = question.split("3. ")[0]
            formatted_questions.append(question[2:].strip())
    return formatted_questions

def get_mcqs_from_docs_for_pdf(docs, num_questions):
    all_mcqs = []
    random.shuffle(docs)
    i = 0
    while i < len(docs) and len(all_mcqs) < num_questions:
        doc = docs[i]
        # print(f"Processing document chunk: {doc.page_content[:100]}...")  
        mcqs = generate_mcqs(doc.page_content, num_questions)
        print(f"Generated {len(mcqs)} MCQs from a chunk")
        all_mcqs.extend(mcqs)
        i += 1
    random.shuffle(all_mcqs)
    return all_mcqs[:num_questions]

def get_mcqs_from_docs_for_text(docs, num_questions):
    all_mcqs = []
    random.shuffle(docs)
    i = 0
    while i < len(docs) and len(all_mcqs) < num_questions:
        doc = docs[i]
        # print(f"Processing document chunk: {doc.page_content[:100]}...")  
        mcqs = generate_mcqs(doc['page_content'], num_questions)
        print(f"Generated {len(mcqs)} MCQs from a chunk")
        all_mcqs.extend(mcqs)
        i += 1
    random.shuffle(all_mcqs)
    return all_mcqs[:num_questions]

def generate_mcqs_interface_pdf(pdf_path, num_questions):
    if num_questions <= 0:
        return "Number of questions must be a positive integer."
    docs, _ = process_pdf(pdf_path)
    mcqs = get_mcqs_from_docs_for_pdf(docs, num_questions)
    output = ""
    for i, mcq in enumerate(mcqs):
        output += "Question " + str(i + 1) + ": " + mcq + "\n\n"
    return output

def generate_mcqs_interface_text(text, num_questions):
    if num_questions <= 0:
        return "Number of questions must be a positive integer."
    docs, _ = process_text(text)
    mcqs = get_mcqs_from_docs_for_text(docs, num_questions)
    output = ""
    for i, mcq in enumerate(mcqs):
        output += "Question " + str(i + 1) + ": " + mcq + "\n\n"
    return output

# Gradio app
with gr.Blocks() as iface:
    gr.Markdown("# MCQ Generator")
    with gr.Tabs():
        with gr.TabItem("Upload PDF"):
            pdf_input = gr.File(label="Upload PDF", type="filepath")
            num_questions_pdf = gr.Number(label="Number of Questions", value=5, precision=0)
            pdf_output = gr.Textbox(label="Generated MCQs")
            pdf_button = gr.Button("Generate MCQs")
            pdf_button.click(generate_mcqs_interface_pdf, inputs=[pdf_input, num_questions_pdf], outputs=pdf_output)
        
        with gr.TabItem("Input Text"):
            text_input = gr.Textbox(label="Enter Text", lines=10)
            num_questions_text = gr.Number(label="Number of Questions", value=5, precision=0)
            text_output = gr.Textbox(label="Generated MCQs")
            text_button = gr.Button("Generate MCQs")
            text_button.click(generate_mcqs_interface_text, inputs=[text_input, num_questions_text], outputs=text_output)

iface.launch()
