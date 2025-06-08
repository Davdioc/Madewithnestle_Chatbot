import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Azure OpenAI Embeddings Configuration
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDINGS_API = os.getenv("AZURE_OPENAI_EMBEDDINGS_API")
    AZURE_OPENAI_EMBEDDINGS_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
    
    # Google Maps API Configuration
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    
    # CORS Configuration
    ALLOWED_ORIGINS = [
        "https://yellow-beach-082df8c0f.6.azurestaticapps.net",
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"   
    ]