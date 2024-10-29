from docx import Document
from pydantic.v1 import BaseModel, Field
from crewai_tools import BaseTool
from typing import Type

class FileReadToolResult(BaseModel):
    file_content: str

class FileReadToolInput(BaseModel):
    """Input for FileReadTool."""
    path: str = Field(..., description="The path to the document.")


class FileReadTool(BaseTool):
    name: str = "File Read Tool"
    description: str = "Extract the content from the given document."
    args_schema: Type[BaseModel] = FileReadToolInput

    def _run(self, path:str) -> FileReadToolResult:
        
        try:
            doc = Document(path)
            for paragraph in doc.paragraphs:
                content = paragraph.text
        except Exception as e:
            print(f"Error reading the .docx file: {e}")

            
        return content
    
