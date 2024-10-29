import os
#from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from joblib import load
from pydantic.v1 import BaseModel, Field
from crewai_tools import BaseTool
from typing import List, Type
import re
from docx2python import docx2python
from dotenv import load_dotenv
load_dotenv()


class TFIDFActivityRecommendationResult(BaseModel):
    file_name: str
    score: float
    topic: str
    cont: str

class TFIDFActivityRecommendationToolInput(BaseModel):
    """Input for ActivityRecommendationTool."""
    background: str = Field(..., description="The query describing user's background or interests.")


class TFIDFActivityRecommendationTool(BaseTool):
    name: str = "ERF Activity Recommendation"
    description: str = "Search for top 3 activities that best fit the information in the query."
    args_schema: Type[BaseModel] = TFIDFActivityRecommendationToolInput

    def _run(self, background:str) -> List[TFIDFActivityRecommendationResult]:
        def top_documents(background, vectorizer, tfidf_matrix, filenames, threshold=0.01):
            query_vec = vectorizer.transform([background])
            cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.where(cosine_similarities >= threshold)[0]
            sorted_indices = top_indices[np.argsort(cosine_similarities[top_indices])[::-1]]
            top_files_with_scores = [(filenames[i], cosine_similarities[i]) for i in sorted_indices[:3]]
        
            return top_files_with_scores
        
        vectorizer, tfidf_matrix, filenames = load('data.joblib')
        query_vec = vectorizer.transform([background])
        top = top_documents(background, vectorizer, tfidf_matrix, filenames)

        # extract each part
        contents = {}
        topics = {}
        folder_path = 'synthetic-documents'
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if filename.endswith('.docx'):
                with docx2python(file_path) as docx_content:
                    content = docx_content.text
                    contents[filename] = content
                    
                match = re.search(r'Title:(.*?)(?=Topic Categorization:)', content, re.DOTALL)
                if match:
                    topic = match.group(1).strip()
                    topics[filename] = topic
            
        output = []
        
        for file_and_score in top:
            filename, score = file_and_score
            file_index = filenames.index(filename)
            doc_vec = tfidf_matrix[file_index]
            
            output.append(TFIDFActivityRecommendationResult(
                file_name="synthetic-documents/"+filename,
                score=round(score, 2),
                topic=topics[filename],
                cont=contents[filename]
            ))

        return output
    
