import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from joblib import dump, load

def get_matrix(folder_path):
    def read_docx(file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    
    documents = []  # Store the content of each doc (e.g. [doc1, doc2, doc3])
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            doc_content = read_docx(file_path)
            documents.append(doc_content)
            filenames.append(filename)  # Store the filename
            
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    dump((vectorizer, tfidf_matrix, filenames), 'data.joblib')
    print('Matrix saved!')
