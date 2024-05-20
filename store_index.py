from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from langchain.vectorstores import Chroma

persist_directory = './db'

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=persist_directory)
