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