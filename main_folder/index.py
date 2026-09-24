import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
PDF_PATH = os.path.join(BASE_DIR, "GRU.pdf")

# Load API key
load_dotenv(DOTENV_PATH)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Check the .env file in the main_folder directory."
    )

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Number of pages:", len(documents))

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print("Number of chunks:", len(chunks))

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-2.5-flash", google_api_key=api_key
)

# Build FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("gru_index")

print(" FAISS index built and saved!")
