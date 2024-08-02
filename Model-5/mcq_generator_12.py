#mcq generator with improvised prompt and download options for output, corrected mcq generation logic for raw text -  updated from mcq_generator_11.py
import re
import tempfile
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
from fpdf import FPDF
from docx import Document as DocxDocument
from io import BytesIO
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
        """I will provide a context and will mention number of questions to generate and you would behave as a strict MCQ generator(stick to context and rules that I specify in this prompt strictly) with one correct option and remaining three options should be wrong. The questions should not just test the comprehension of the candidate rather should also test his/her knowledge, conceptual understanding and reasoning ability...Options as well should be framed in such a way...
            
            The question should be framed should be according to the question type mentioned by me... If not possible to frame according to the question type mentioned, its okay to frame the question belonging to any other type as long as its a correct question based on the context that can be used to test the ability of a candidate...
                 
            Avoid any additional warnings, apologies or any such statements from your side... The template of your response should be as simple as I have mentioned below under "Format the question and options like this:"
        
        Here are examples for each question type:

        1. Conceptual Type
        Context: Mitochondria are often referred to as the "powerhouses" of eukaryotic cells because they play a critical role in energy production. They convert nutrients into adenosine triphosphate (ATP), the energy currency of the cell, through a process known as cellular respiration. Understanding this function is crucial for grasping how cells sustain their activities and support various bodily functions.
        Example Question: What is the primary function of mitochondria in eukaryotic cells?
        a) Photosynthesis
        b) Cellular respiration
        c) Protein synthesis
        d) DNA replication
        Answer: b) Cellular respiration

        2. Numerical Type
        Context: Calculating average speed is a fundamental concept in physics and everyday problem-solving. When analyzing how fast a vehicle is moving, knowing the distance traveled and the time taken allows for the calculation of average speed, which is essential for understanding vehicle performance and making informed decisions about travel and transportation.
        Example Question: If a car travels 60 miles in 1 hour, what is its average speed in miles per hour?
        a) 50 mph
        b) 60 mph
        c) 70 mph
        d) 80 mph
        Answer: b) 60 mph

        3. Factual Type
        Context: Historical knowledge is fundamental to understanding the development of nations and their political systems. The first President of the United States, George Washington, played a pivotal role in the founding of the country and set many precedents for the office that followed. This fact is a cornerstone of American history and civic education.
        Example Question: Who was the first President of the United States?
        a) Thomas Jefferson
        b) Abraham Lincoln
        c) George Washington
        d) John Adams
        Answer: c) George Washington

        4. Analytical Type
        Context: In economics, understanding the relationship between price and quantity demanded helps in analyzing market behaviors and making predictions. The inverse relationship between the price of a good and the quantity demanded is a fundamental concept that reflects consumer behavior and market dynamics.
        Example Question: Given that an increase in the price of a good leads to a decrease in its quantity demanded, what type of relationship does this demonstrate?
        a) Direct relationship
        b) Inverse relationship
        c) No relationship
        d) Complex relationship
        Answer: b) Inverse relationship

        5. Comparative Type
        Context: The distinction between prokaryotic and eukaryotic cells is a basic concept in biology. Prokaryotic cells, such as bacteria, lack a defined nucleus, whereas eukaryotic cells, including those in plants and animals, have a well-defined nucleus. Understanding these differences is essential for studying cellular biology and the diversity of life.
        Example Question: Which of the following is a key difference between prokaryotic and eukaryotic cells?
        a) Prokaryotic cells have a nucleus, while eukaryotic cells do not.
        b) Eukaryotic cells have a nucleus, while prokaryotic cells do not.
        c) Prokaryotic cells have a cell membrane, while eukaryotic cells do not.
        d) Both cell types have the same structures.
        Answer: b) Eukaryotic cells have a nucleus, while prokaryotic cells do not.

        6. Definition Type
        Context: Photosynthesis is a crucial process for life on Earth, allowing plants to convert sunlight into chemical energy. This process is fundamental to plant biology and has significant implications for ecosystems and the environment, as it produces oxygen and serves as the basis of the food chain.
        Example Question: What is the definition of "photosynthesis"?
        a) The process by which plants absorb nutrients from soil
        b) The process by which plants convert sunlight into chemical energy
        c) The process by which plants release oxygen into the atmosphere
        d) The process by which plants grow in the absence of light
        Answer: b) The process by which plants convert sunlight into chemical energy

        7. Application Type
        Context: Scaling recipes is a common task in cooking and baking, especially when preparing food for different numbers of servings. By applying proportions accurately, one ensures that the recipe yields the correct amount of food and maintains the intended taste and texture.
        Example Question: If a recipe requires 2 cups of flour for 4 servings, how many cups of flour are needed for 10 servings?
        a) 4 cups
        b) 5 cups
        c) 6 cups
        d) 8 cups
        Answer: b) 5 cups

        8. Inference Type
        Context: Inferring skill levels based on experience is a common practice in many fields. For instance, a violinist who has been practicing for several years and performs at various events is likely to have a high level of expertise, reflecting their dedication and experience.
        Example Question: Based on the fact that a person has been practicing violin for several years and performs at various events, what can be inferred about their skill level?
        a) They are a beginner violinist.
        b) They have a moderate skill level.
        c) They are an expert violinist.
        d) They have never played the violin before.
        Answer: c) They are an expert violinist.

        9. Critical Type
        Context: Analyzing the arguments surrounding economic policies, such as minimum wage increases, requires critical thinking. Evaluating the assumptions and potential outcomes helps in understanding the broader implications of such policies and their impact on different stakeholders.
        Example Question: What is a major critique of the argument that increasing the minimum wage will always lead to higher unemployment?
        a) It ignores potential benefits to workers.
        b) It assumes all businesses are equally affected.
        c) It fails to consider regional economic differences.
        d) All of the above
        Answer: d) All of the above

        10. Sequence Type
        Context: The scientific method is a systematic approach used in scientific research to ensure that investigations are conducted in a logical and consistent manner. Understanding the correct sequence of steps helps in designing experiments and drawing valid conclusions based on empirical evidence.
        Example Question: What is the correct sequence of steps in the scientific method?
        a) Hypothesis, Experiment, Observation, Conclusion
        b) Observation, Hypothesis, Experiment, Conclusion
        c) Experiment, Conclusion, Hypothesis, Observation
        d) Conclusion, Observation, Experiment, Hypothesis
        Answer: b) Observation, Hypothesis, Experiment, Conclusion"""

        f"Generate {num_questions} multiple-choice {qtype} questions from the following given text:\n\n"
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
    print("Response Text:", response_text)
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

def get_mcqs_from_docs(docs, num_questions, exclude_indices, pdf_upload = True):
    all_mcqs = []
    if pdf_upload:
        filtered_docs = [doc.page_content for idx, doc in enumerate(docs) if idx not in exclude_indices]
    else:
        filtered_docs = [doc['page_content'] for idx, doc in enumerate(docs) if idx not in exclude_indices]
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

def update_chunks_multiple_pdfs(pdf_files, page_ranges_list):
    all_docs = []
    total_pages_list = []
    for pdf_file, page_ranges in zip(pdf_files, page_ranges_list):
        docs, _, total_pages = process_pdf_with_pypdf2(pdf_file, page_ranges)
        total_pages_list.append(total_pages)
        all_docs.extend(docs)
    return all_docs, total_pages_list

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
    mcqs = get_mcqs_from_docs(docs, num_questions, [], False)
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

def create_txt(content):
    with open("output.txt", "w") as file:
        file.write(content)
    with open("output.txt", "r") as file:
        st.download_button(
            label="Download Text File",
            data=file,
            file_name="output.txt",
            mime="text/plain"
        )

def create_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf_output_path = "output.pdf"
    pdf.output(pdf_output_path)
    with open(pdf_output_path, "rb") as file:
        st.download_button(
            label="Download PDF File",
            data=file,
            file_name="output.pdf",
            mime="application/pdf"
        )

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
                content = st.session_state.mcqs_with_answers
            else:
                st.text_area("Generated MCQs \n", "".join(st.session_state.mcqs_without_answers), height=300)
                content = st.session_state.mcqs_without_answers
            if st.button("Generate TXT"):
                create_txt(content)

            if st.button("Generate PDF"):
                create_pdf(content)

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
            st.text_area("Generated MCQs", "".join(st.session_state.mcqs_with_answers), height=300)
            content = st.session_state.mcqs_with_answers
        else:
            st.text_area("Generated MCQs", "".join(st.session_state.mcqs_without_answers), height=300)
            content = st.session_state.mcqs_without_answers
        
        if st.button("Generate TXT"):
            create_txt(content)

        if st.button("Generate PDF"):
            create_pdf(content)
