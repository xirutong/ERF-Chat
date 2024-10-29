import os
from docx import Document
import joblib
from pydantic.v1 import BaseModel, Field
from crewai_tools import BaseTool
from typing import List, Type
import re
from docx2python import docx2python
from sentence_transformers import SentenceTransformer, util

class SBERTActivityRecommendationResult(BaseModel):
    file_name: str
    score: float
    topic: str
    cont: str

class SBERTActivityRecommendationToolInput(BaseModel):
    """Input for ActivityRecommendationTool."""
    background: str = Field(..., description="The query describing user's background or interests.")


class SBERTActivityRecommendationTool(BaseTool):
    name: str = "ERF Activity Recommendation"
    description: str = "Search for top 3 activities that best fit the information in the query."
    args_schema: Type[BaseModel] = SBERTActivityRecommendationToolInput
    
    def _run(self, background:str) -> List[SBERTActivityRecommendationResult]:
        folder_path = 'synthetic-documents'
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        query_embedding = model.encode(background)
        document_embeddings = joblib.load("embeddings.joblib")
        cosine_similarities = util.cos_sim(query_embedding, document_embeddings)
        
        filenames = []
    
        for filename in os.listdir(folder_path):
            if filename.endswith(".docx"):
                filenames.append(filename)
                
        similarity_dict = {}
        for i, similarity in enumerate(cosine_similarities[0]):
            similarity_dict[filenames[i]] = f"{similarity:.4f}"
        
        def get_top_n_files(similarity_dict, n=3):
            similarity_dict = {k: float(v) for k, v in similarity_dict.items()}
            sorted_files = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
            top_n_files = sorted_files[:n]

            return top_n_files
        
        file_and_score=get_top_n_files(similarity_dict)
        
        
        contents = {}
        topics = {}
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if filename.endswith('.docx'):
                with docx2python(file_path) as docx_content:
                    content = docx_content.text
                    contents[filename] = content
                
                """match = re.search(r'Brief Description:(.*?)(?=Highlights:)', content, re.DOTALL)
                if match:
                    brief_description = match.group(1).strip()
                    brief_descriptions[filename] = brief_description"""
                    
                match = re.search(r'Title:(.*?)(?=Topic Categorization:)', content, re.DOTALL)
                if match:
                    topic = match.group(1).strip()
                    topics[filename] = topic
            
            
        output = []
        
        for filename, score in file_and_score:
            output.append(SBERTActivityRecommendationResult(
                file_name="synthetic-documents/"+filename,
                score=round(score, 2),
                topic=topics[filename],
                cont=contents[filename],
            ))
            
        return output
    
