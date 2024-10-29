from sentence_transformers import SentenceTransformer, util
from docx import Document
import os
import joblib

def train(folder_path:str):
        # Load pre-trained Sentence-BERT model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Collect all the docs
        def read_docx(file_path):
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
    
        documents = []  # Store the content of each doc (e.g. [doc1, doc2, doc3])
    
        for filename in os.listdir(folder_path):
            if filename.endswith(".docx"):
                file_path = os.path.join(folder_path, filename)
                doc_content = read_docx(file_path)
                documents.append(doc_content)
        
        # Encode the documents and the query into embeddings        
        document_embeddings = model.encode(documents)
        
        joblib.dump(document_embeddings, 'embeddings.joblib')
        print('Embeddings saved!')