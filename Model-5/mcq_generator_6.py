#mcq generator with 3 pdfs as input with page and chunk filtering
import re
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
locale.getpreferredencoding()

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

def process_pdf(pdf_path, page_ranges=None):
    try:
        loader = PyPDFLoader(pdf_path, extract_images=True)
        pages = loader.load()
        
        # Handling page ranges
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
        
        # Splitting text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        docs = text_splitter.split_documents(pages)

        # Creating a vector database
        vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)
        
        return docs, vector_db, len(pages)
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return [], None, 0

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

def get_mcqs_from_docs(docs, num_questions, pdf_input, exclude_indices):
    all_mcqs = []
    filtered_docs = [doc for idx, doc in enumerate(docs) if idx not in exclude_indices]
    random.shuffle(filtered_docs)
    i = 0
    while i < len(filtered_docs) and len(all_mcqs) < num_questions:
        doc = filtered_docs[i]
        if pdf_input:
            mcqs = generate_mcqs(doc.page_content, num_questions)
        else:
            mcqs = generate_mcqs(doc['page_content'], num_questions)
        print(f"Generated {len(mcqs)} MCQs from a chunk")
        all_mcqs.extend(mcqs)
        i += 1
    random.shuffle(all_mcqs)
    return all_mcqs[:num_questions]

def update_chunks_multiple_pdfs(pdf_paths, page_ranges_list):
    all_docs = []
    total_pages_list = []

    for pdf_path, page_ranges in zip(pdf_paths, page_ranges_list):
        docs, _, total_pages = process_pdf(pdf_path, page_ranges)
        total_pages_list.append(total_pages)
        all_docs.extend(docs)

    return all_docs, total_pages_list

def generate_mcqs_from_multiple_pdfs(pdf_paths, page_ranges_list, num_questions, exclude_indices):
    if not pdf_paths:
        return "No PDFs uploaded."
    
    all_docs, _ = update_chunks_multiple_pdfs(pdf_paths, page_ranges_list)
    if not all_docs:
        return "Error processing PDFs."

    all_mcqs = get_mcqs_from_docs(all_docs, num_questions, True, exclude_indices)
    output = ""
    for i, mcq in enumerate(all_mcqs):
        output += f"Question {i + 1}: {mcq}\n\n"
    return output

def generate_mcqs_interface_text(text, num_questions):
    if num_questions <= 0:
        return "Number of questions must be a positive integer."
    docs, _ = process_text(text)
    mcqs = get_mcqs_from_docs(docs, num_questions, False, [])
    output = ""
    for i, mcq in enumerate(mcqs):
        output += "Question " + str(i + 1) + ": " + mcq + "\n\n"
    return output

with gr.Blocks() as iface:
    gr.Markdown("# MCQ Generator")

    with gr.Tabs():
        with gr.TabItem("Upload PDFs"):
            pdf_inputs = [gr.File(label=f"Upload PDF {i+1}", type="filepath") for i in range(3)]
            total_pages_texts = [gr.Textbox(label=f"Total Pages of PDF {i+1}", interactive=False) for i in range(3)]
            page_ranges_inputs = [gr.Textbox(label=f"Page Ranges of PDF {i+1}", placeholder="Enter page ranges") for i in range(3)]
            display_chunks_button = gr.Button("Display Chunks for Filtering")
            chunk_selection = gr.CheckboxGroup(label="Select Chunks to Remove", visible = True)
            num_questions_pdf = gr.Number(label="Number of Questions", value=5, precision=0)
            generate_button = gr.Button("Generate MCQs")
            pdf_output = gr.Textbox(label="Generated MCQs")

            def show_total_pages_for_all_pdfs(*pdf_files):
                total_pages_list = []
                for pdf_file in pdf_files:
                    if pdf_file:
                        _, _, total_pages = process_pdf(pdf_file.name)
                        total_pages_list.append(total_pages)
                return total_pages_list

            def generate_mcqs_for_all_pdfs(pdf_files, page_ranges_list, num_questions, chunk_exclusions):
                pdf_paths = [pdf_file.name for pdf_file in pdf_files if pdf_file]
                return generate_mcqs_from_multiple_pdfs(pdf_paths, page_ranges_list, num_questions, chunk_exclusions)

            def update_total_pages(*pdf_files):
                total_pages_list = show_total_pages_for_all_pdfs(*pdf_files)
                return total_pages_list

            for pdf_input, total_pages_text in zip(pdf_inputs, total_pages_texts):
                pdf_input.change(
                    update_total_pages,
                    inputs=[pdf_input],
                    outputs=[total_pages_text]
                )

            def on_display_chunks_button_click(*args):
                num_pdfs = 3  
                pdf_files = args[:num_pdfs]
                page_ranges_inputs = args[num_pdfs:num_pdfs*2]
            
                if len(page_ranges_inputs) < num_pdfs:
                    return "Error: Not enough page ranges provided for the number of PDFs."

                pdf_page_ranges = page_ranges_inputs[:num_pdfs]
                
                all_docs = []
                for i, pdf_file_path in enumerate(pdf_files):
                    if pdf_file_path:
                        page_range = pdf_page_ranges[i] if i < len(pdf_page_ranges) else None
                        
                        try:
                            docs, _, _ = process_pdf(pdf_file_path, page_range)
                            all_docs.extend(docs)
                        except Exception as e:
                            print(f"Error processing PDF {pdf_file_path} with page range {page_range}: {e}")
 
                chunk_ids = [f"Chunk {i+1}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)]

                return gr.update(visible=True, choices=chunk_ids)

            display_chunks_button.click(
                on_display_chunks_button_click,
                inputs=[*pdf_inputs, *page_ranges_inputs],
                outputs=[chunk_selection]
            )

            def extract_chunk_indices(exclusion_list):
                indices = []
                for item in exclusion_list:
                    match = re.search(r'(\d+)', item)  
                    if match:
                        index = int(match.group(1)) - 1  
                        indices.append(index)
                return indices

            def on_generate_mcqs_button_click(*args):
                pdf_files = args[:3]
                page_ranges_list = args[3:6]
                num_questions = args[6]
                chunk_exclusions_raw = args[7]

                chunk_exclusions = extract_chunk_indices(chunk_exclusions_raw)

                return generate_mcqs_for_all_pdfs(pdf_files, page_ranges_list, num_questions, chunk_exclusions)

            generate_button.click(
                on_generate_mcqs_button_click,
                inputs=[*pdf_inputs, *page_ranges_inputs, num_questions_pdf, chunk_selection],
                outputs=[pdf_output]
            )

        with gr.TabItem("Text Input"):
            text_input = gr.Textbox(label="Input Text", placeholder="Enter text here")
            num_questions_text = gr.Number(label="Number of Questions", value=5, precision=0)
            text_output = gr.Textbox(label="Generated MCQs")

            def generate_mcqs_interface_text(text, num_questions):
                if num_questions <= 0:
                    return "Number of questions must be a positive integer."
                docs, _ = process_text(text)
                mcqs = get_mcqs_from_docs(docs, num_questions, False, [])
                output = ""
                for i, mcq in enumerate(mcqs):
                    output += "Question " + str(i + 1) + ": " + mcq + "\n\n"
                return output

            text_input.submit(
                generate_mcqs_interface_text,
                inputs=[text_input, num_questions_text],
                outputs=[text_output]
            )

iface.launch()
