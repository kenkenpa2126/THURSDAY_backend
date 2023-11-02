from pydantic import BaseModel
from typing import List

class ReadDocs(BaseModel):
    doc_path: str

class StorePDF(BaseModel):
    pdf: str

class StoreText(BaseModel):
    text: str

class AskAgent(BaseModel):
    message: str