from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ChatRequest(BaseModel):
    question: str
    name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    answer: str

class AddNode(BaseModel):
    text: str

class AddNodeResponse(BaseModel):
    status: str = Field(default="success", description="Status of the operation")