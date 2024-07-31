# improved prompt for mcq generator and option to hide or show answers - updated from mcq_generator_10.py
import re
import locale
import weaviate
import random
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from secret import WEAVIATE_CLUSTER, WEAVIATE_API_KEY, HUGGING_FACE_API_TOKEN

# Setting encoding
locale.getpreferredencoding()

# Initializing Weaviate client
client = weaviate.Client(
    url=WEAVIATE_CLUSTER,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

# Defining the embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Initializing HuggingFaceHub model
model = HuggingFaceHub(
    huggingfacehub_api_token=HUGGING_FACE_API_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 1000}
)

# Function to process PDF
def process_pdf_with_pypdf2(pdf_file, page_ranges=None):
    try:
        pdf_reader = PdfReader(pdf_file)
        pages = []
        for i in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
                else:
                    st.warning(f"Page {i + 1} contains no text or could not be extracted.")
            except Exception as e:
                st.warning(f"Error extracting text from page {i + 1}: {e}")

        if page_ranges:
            all_pages = []
            ranges = page_ranges.split(',')
            for page_range in ranges:
                page_range = page_range.strip()
                if '-' in page_range:
                    start, end = map(int, page_range.split('-'))
                    all_pages.extend(pages[start - 1:end])
                else:
                    page_number = int(page_range)
                    all_pages.append(pages[page_number - 1])
            pages = all_pages

        docs = [Document(page_content=page) for page in pages]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        split_docs = text_splitter.split_documents(docs)

        vector_db = Weaviate.from_documents(split_docs, embeddings, client=client, by_text=False)

        return split_docs, vector_db, len(pages)
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return [], None, 0

# Function to generate MCQs
def generate_mcqs(text_chunk, num_questions):
    question_types = [
        'conceptual', 'numerical', 'factual', 'analytical', 'comparative', 
        'definition', 'application', 'inference', 'critical', 'sequence'
    ]
    qtype = random.choice(question_types)
    print("Qtype :", qtype)
    prompt = (
        f"Generate {num_questions} multiple-choice {qtype} questions from the following text:\n\n"
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
    response_text = response.split("Questions:")[-1]
    start_index = response_text.find("1.")
    response_text = response_text[start_index:]
    questions = response_text.split("\n\n")
    formatted_questions = []
    for question in questions:
        answer_index = question.find("Answer: ")
        question = question[:answer_index + 10]
        if len(question) == answer_index + 9:
            question += ")"
        if len(question) == answer_index + 10 and all(opt in question for opt in ['a)', 'b)', 'c)', 'd)', 'Answer:']):
            if "1." in question:
                question = question.split("2. ")[0]
            elif "2." in question:
                question = question.split("3. ")[0]
            formatted_questions.append(question[2:].strip())
    return formatted_questions

# Function to get MCQs from documents
def get_mcqs_from_docs(docs, num_questions, exclude_indices):
    all_mcqs = []
    filtered_docs = [doc.page_content for idx, doc in enumerate(docs) if idx not in exclude_indices]
    random.shuffle(filtered_docs)
    i = 0
    while i < len(filtered_docs) and len(all_mcqs) < num_questions:
        doc = filtered_docs[i]
        mcqs = generate_mcqs(doc, num_questions)
        st.write(f"Generated {len(mcqs)} MCQs from a chunk")
        all_mcqs.extend(mcqs)
        i += 1
    random.shuffle(all_mcqs)
    return all_mcqs[:num_questions]


# Function to update chunks from multiple PDFs
def update_chunks_multiple_pdfs(pdf_files, page_ranges_list):
    all_docs = []
    total_pages_list = []
    for pdf_file, page_ranges in zip(pdf_files, page_ranges_list):
        docs, _, total_pages = process_pdf_with_pypdf2(pdf_file, page_ranges)
        total_pages_list.append(total_pages)
        all_docs.extend(docs)
    return all_docs, total_pages_list

# Function to generate MCQs from multiple PDFs
def generate_mcqs_from_multiple_pdfs(pdf_files, page_ranges_list, num_questions, exclude_indices):
    if exclude_indices is None:
        exclude_indices = []

    if not pdf_files:
        return "No PDFs uploaded."
    
    all_docs, _ = update_chunks_multiple_pdfs(pdf_files, page_ranges_list)
    if not all_docs:
        return "Error processing PDFs."

    all_mcqs = get_mcqs_from_docs(all_docs, num_questions, exclude_indices)
    mcqs_without_answers = ""
    output = ""
    for i, mcq in enumerate(all_mcqs):
        answer_index = mcq.find("Answer: ")
        output += f"Question {i + 1}: {mcq}\n\n"
        mcqs_without_answers += f"Question {i + 1}: {mcq[:answer_index]}\n\n"
    return output, mcqs_without_answers

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
    
# Function to generate MCQs from text
def generate_mcqs_interface_text(text, num_questions):
    if num_questions <= 0:
        return "Number of questions must be a positive integer."
    docs, _ = process_text(text)
    mcqs = get_mcqs_from_docs(docs, num_questions, [])
    output = ""
    for i, mcq in enumerate(mcqs):
        output += f"Question {i + 1}: {mcq}\n\n"
    mcqs_without_answers = ""
    output = ""
    for i, mcq in enumerate(mcqs):
        answer_index = mcq.find("Answer: ")
        output += f"Question {i + 1}: {mcq}\n\n"
        mcqs_without_answers += f"Question {i + 1}: {mcq[:answer_index]}\n\n"
    return output, mcqs_without_answers

# Streamlit UI
st.title("MCQ Generator")

tab = st.selectbox("Select a tab", ["Upload PDFs", "Text Input"])

if tab == "Upload PDFs":
    num_pdfs = st.number_input("Number of PDFs", min_value=1, max_value=10, value=3)
    
    pdf_files = [st.file_uploader(f"Upload PDF {i+1}", type="pdf") for i in range(num_pdfs)]
    
    if all(pdf_files):
        if st.button("Display Total Pages"):
            if 'total_pages' not in st.session_state:
                st.session_state.total_pages = {}
            
            if all(pdf_files):
                for i, pdf_file in enumerate(pdf_files):
                    if pdf_file:
                        _, _, total_pages_count = process_pdf_with_pypdf2(pdf_file)
                        st.session_state.total_pages[f"PDF {i+1}"] = total_pages_count

        if 'total_pages' in st.session_state:
            for pdf_name, total_pages_count in st.session_state.total_pages.items():
                st.write(f"Total Pages of {pdf_name}: {total_pages_count}")

        page_ranges_list = [st.text_input(f"Page Ranges of PDF {i+1}", placeholder="Enter page ranges") for i in range(num_pdfs)]

        # Initializing exclude_indices and all_docs in session state
        if 'exclude_indices' not in st.session_state:
            st.session_state.exclude_indices = []

        if 'all_docs' not in st.session_state:
            st.session_state.all_docs = []

        if 'chunks_visible' not in st.session_state:
            st.session_state.chunks_visible = False

        # Displaying and filtering chunks
        if st.button("Display Chunks for Filtering", key="display_chunks_for_filtering"):
            if 'all_docs' not in st.session_state or not st.session_state.all_docs:
                all_docs = []
                for pdf_file, page_ranges in zip(pdf_files, page_ranges_list):
                    docs, _, _ = process_pdf_with_pypdf2(pdf_file, page_ranges)
                    all_docs.extend(docs)
                st.session_state.all_docs = all_docs

            if st.session_state.all_docs:
                st.session_state.exclude_indices = [] 

                st.session_state.chunks_visible = True

        if st.session_state.chunks_visible:
            st.write("Displaying Chunks:")
            chunks = [doc.page_content for doc in st.session_state.all_docs]

            for i, chunk in enumerate(chunks):
                st.write(f"**Chunk {i + 1}:**")
                st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk) 

                # Checkbox for excluding chunks
                if st.checkbox(f"Exclude Chunk {i + 1}", key=f"exclude_chunk_{i}", value=i in st.session_state.exclude_indices):
                    if i not in st.session_state.exclude_indices:
                        st.session_state.exclude_indices.append(i)
                else:
                    if i in st.session_state.exclude_indices:
                        st.session_state.exclude_indices.remove(i)

            if st.button("Done Excluding", key="done_excluding"):
                st.session_state.chunks_visible = False
                st.write("Excluding done. Now you can generate MCQs.")

        if not st.session_state.chunks_visible and st.session_state.exclude_indices:
            marked_chunks = [i + 1 for i in st.session_state.exclude_indices]
            st.write(f"Excluded Chunks: {marked_chunks}")

        num_questions = st.number_input("Number of Questions", min_value=1, value=5)
        show_answers = st.checkbox("Show Answers", value=True)

        if st.button("Generate MCQs"):
            if pdf_files:
                with st.spinner("Processing PDFs and generating MCQs..."):
                    mcqs_with_answers, mcqs_without_answers = generate_mcqs_from_multiple_pdfs([file for file in pdf_files], page_ranges_list, num_questions, st.session_state.exclude_indices)
                    st.session_state.mcqs_with_answers = mcqs_with_answers
                    st.session_state.mcqs_without_answers = mcqs_without_answers
        
        if 'mcqs_with_answers' in st.session_state and 'mcqs_without_answers' in st.session_state:
            if show_answers:
                st.text_area("Generated MCQs \n", "".join(st.session_state.mcqs_with_answers), height=300)
            else:
                st.text_area("Generated MCQs \n", "".join(st.session_state.mcqs_without_answers), height=300)

                    
elif tab == "Text Input":
    text = st.text_area("Enter text here", height=300)
    num_questions = st.number_input("Number of Questions", min_value=1, value=5)
    show_answers = st.checkbox("Show Answers", value=True)

    if st.button("Generate MCQs"):
        if text:
            mcqs_with_answers, mcqs_without_answers = generate_mcqs_interface_text(text, num_questions)
            st.session_state.mcqs_with_answers = mcqs_with_answers
            st.session_state.mcqs_without_answers = mcqs_without_answers

    if 'mcqs_with_answers' in st.session_state and 'mcqs_without_answers' in st.session_state:
        if show_answers:
            st.text_area("Generated MCQs", "\n\n".join(st.session_state.mcqs_with_answers), height=300)
        else:
            st.text_area("Generated MCQs", "\n\n".join(st.session_state.mcqs_without_answers), height=300)