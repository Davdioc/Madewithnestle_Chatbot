from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import re
import requests
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize app
app = FastAPI(title="Made with Nestlé Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yellow-beach-082df8c0f.6.azurestaticapps.net",
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"   
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    answer: str

# Cache for compiled regex patterns
_lucene_chars_pattern = re.compile(r'[+\-&|!(){}[\]^"~*?:\\/]')
_whitespace_pattern = re.compile(r'\s+')
_text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Global variables for components
graph = None
vector_retriever = None
agent_executor = None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    r = 6371 
    distance = 2 * r * atan2(sqrt(d), sqrt(1 - d))
    return distance

@tool
def search_nearby_locations(product_name: str, user_lat: float, user_lng: float, radius: int = 10500) -> str:
    """
    Search for nearby locations that sell a specific Nestlé product using Google Places API.
    
    Args:
        product_name: Name of the Nestlé product to search for
        user_lat: User's latitude coordinate  
        user_lng: User's longitude coordinate
        radius: Search radius in meters (default 10500m = ~10.5km)
    
    Returns:
        Formatted string with nearby locations, addresses, distances, and status
    """
    try:
        API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
        if not API_KEY:
            return "Google Maps API key not configured."
            
        endpoint_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        
        params = {
            'keyword': f"{product_name} nestlé",
            'location': f'{user_lat},{user_lng}',
            'radius': radius,
            'type': 'store',
            'key': API_KEY
        }

        response = requests.get(endpoint_url, params=params)
        
        if response.status_code != 200:
            return f"Error {response.status_code} while searching for {product_name}: {response.text}"
            
        data = response.json()
        results = data.get('results', [])[:4]  # Limit to top 4 results

        if not results:
            return f"No nearby stores found for {product_name}. You might want to try searching online retailers."

        output = f"**Top locations near you for {product_name.upper()}:**\n\n"
        
        for idx, place in enumerate(results, start=1):
            name = place.get('name', 'N/A')
            address = place.get('vicinity', 'No address found')
            lat2 = place['geometry']['location']['lat']
            lng2 = place['geometry']['location']['lng']
            distance = calculate_distance(user_lat, user_lng, lat2, lng2)

            # Check if place is currently open
            open_now = place.get("opening_hours", {}).get("open_now", None)
            if open_now is True:
                status = '🟢 Open'
            elif open_now is False:
                status = '🔴 Closed'
            else:
                status = '⚪ Hours Unknown'
            
            rating = place.get('rating', 'No rating')
            
            output += (
                f"{idx}. **{name}**\n"
                f"    Address: {address}\n"
                f"    Distance: {distance:.1f} km away\n"
                f"    Rating: {rating}\n"
                f"    Status: {status}\n"
                f"    [View on Google Maps](https://www.google.com/maps/search/?api=1&query={lat2},{lng2})\n\n"
            )
            
        return output.strip()
        
    except Exception as e:
        return f"Error searching for {product_name}: {str(e)}"

@tool
def get_amazon_product_links(product_name: str) -> str:
    """
    Generate Amazon Canada search links for Nestlé products.
    
    Args:
        product_names: List of product names to search for
        
    Returns:
        Formatted string with Amazon search links
    """
    try:
        base_url = "https://www.amazon.ca/s?k="
        output = "** Find these products on Amazon Canada:**\n\n"
        link = f"{base_url}{product_name}"
        output += f"• [**{product_name.upper()}**]({link})\n"

        return output.strip()
        
    except Exception as e:
        return f"Error generating Amazon links: {str(e)}"

@tool  
def search_nestle_knowledge_base(query: str) -> str:
    """
    Search the Nestlé knowledge base for product information, recipes, nutrition facts, and general questions.

    Args:
        query: The search query or question about Nestlé products

    Returns:
        Relevant information from the knowledge base
    """
    try:
        global vector_retriever, graph
        
        if not vector_retriever or not graph:
            return "Knowledge base not available."
            
        vector_results = vector_retriever.invoke(query)
        vector_content = "\n".join([doc.page_content for doc in vector_results])
        graph_content = graph_retriever_simple(query, graph)
        
        combined_content = f"Vector Results:\n{vector_content}\n\nGraph Results:\n{graph_content}"
        print(f"Graph content: {graph_content}\n Vector content: {vector_content}")
        return combined_content if combined_content.strip() else "No relevant information found in the knowledge base."

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

def remove_lucene_chars(text):
    """Clean text for Lucene search"""
    cleaned_text = _lucene_chars_pattern.sub(' ', text)
    cleaned_text = _whitespace_pattern.sub(' ', cleaned_text).strip()
    return cleaned_text

def graph_retriever_simple(question: str, graph: Neo4jGraph) -> str:
    """Simplified graph retriever without entity extraction"""
    try:
        # Extract key terms from the question
        key_terms = [term.strip() for term in question.lower().split() 
                    if len(term.strip()) > 3 and term.strip() not in ['what', 'where', 'when', 'why', 'how', 'the', 'and', 'for', 'with']]
        
        if not key_terms:
            return ""
            
        results = []
        for term in key_terms[:5]:  # Limit to first 3 key terms
            try:
                sanitized_term = remove_lucene_chars(term)
                if sanitized_term:
                    response = graph.query(
                        """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:5})
                        YIELD node,score
                        CALL {
                          WITH node
                          MATCH (node)-[r:!MENTIONS]->(neighbor)
                          RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                          UNION ALL
                          WITH node
                          MATCH (node)<-[r:!MENTIONS]-(neighbor)
                          RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                        }
                        RETURN output LIMIT 20
                        """,
                        {"query": sanitized_term},
                    )
                    results.extend([el['output'] for el in response])
            except Exception:
                continue
                
        return "\n".join(results[:50])  # Limit results
        
    except Exception as e:
        print(f"Error in graph retrieval: {str(e)}")
        return ""
    
def add_to_graph(documents, graph, llm_transformer):
    """Add documents to the graph database"""
    if not documents:
        return
    graph_documents = []
    for doc in documents:
        graph_doc = llm_transformer.convert_to_graph_documents([doc])
        graph_documents.extend(graph_doc)
    if graph_documents:
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        print(f"Added {len(graph_documents)} documents to the graph database.")

def init_components():
    """Initialize all components including the agent"""
    global graph, vector_retriever, agent_executor
    
    # Initialize Neo4j Graph DB connection
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    print("Connected to Neo4j graph database")
    
    # Initialize Azure OpenAI LLM
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.3
    )
    
    # Initialize embeddings and vector store
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_type="azure"
    )
    
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_retriever = vector_index.as_retriever()

    # Define the agent tools
    tools = [
        search_nearby_locations,
        get_amazon_product_links, 
        search_nestle_knowledge_base
    ]
    
    # Create the agent prompt
    system_prompt = """You are Quicky, a helpful AI assistant for the MadeWithNestlé website. Your role is to help users with:

1. **Finding nearby stores** that sell Nestlé products - use the search_nearby_locations tool when users ask "where can I buy...", "where to find...", etc. Do not use this tool unless this user explicitly asks for locations.
2. **Providing product information** - use search_nestle_knowledge_base tool for questions about products, recipes, nutrition, ingredients, etc. ALWAYS add a link of the wource after providing the answer. 
3. **Generating shopping links** - use get_amazon_product_links tool to help users find products online. Always use this when you are asked to locate something
    - a user asks where to buy something,
   - or if physical stores are not found,
   - or if online shopping is mentioned explicitly.
4. **Answering general questions** about Nestlé products, recipes, nutrition, and company information - use the search_nestle_knowledge_base tool to find relevant information.

**Important Guidelines:**
- Always be friendly, helpful, and conversational
- When users ask about locations, ALWAYS use the search_nearby_locations tool if you have their coordinates and get_amazon_product_links regardless of if the coordinates are provided
- For product questions, search the knowledge base first
- Provide comprehensive answers with relevant links and sources
- If you can't find specific information, suggest alternative ways to help
- Always maintain a warm, professional tone as a Nestlé brand representative

**Your name is:** {bot_name}

**Available Tools:**
- search_nearby_locations: Find physical stores near the user
- search_nestle_knowledge_base: Search for product info, recipes, nutrition facts
- get_amazon_product_links: Generate online shopping links

Remember: You're representing the Nestlé brand, so always be helpful, accurate, and brand-appropriate in your responses."""

    # Create the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
    
    print("Agent initialized successfully!")
    return llm

def get_text_chunks_langchain(text):
    """Optimized text chunking using cached splitter"""
    docs = [Document(page_content=x) for x in _text_splitter.split_text(text)]
    return docs

# Initialize components
llm = init_components()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Made with Nestlé Agent Chatbot API is running!"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        global agent_executor
        
        if not agent_executor:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        history_text = ""
        for msg in request.history or []:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        full_input = f"{history_text}User: {request.question} (User location: {request.lat}, {request.lng})" if request.lat and request.lng else f"{history_text} User: {request.question}"
        
        # Execute the agent
        async def run_agent():
            return await asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: agent_executor.invoke({
                    "input": full_input,
                    "bot_name": request.name,
                    "messages": [
                        HumanMessage(content=msg["content"]) if msg["role"] == "user" else SystemMessage(content=msg["content"])
                        for msg in request.history or []
                    ]
                })
            )
        async def process_graph_addition():
            if request.question.strip():
                documents = get_text_chunks_langchain(request.question)
                print("Adding documents to graph...")
                await asyncio.get_event_loop().run_in_executor(
                    executor, add_to_graph, documents, graph, LLMGraphTransformer(llm)
                )

        result = await run_agent()
        # Start graph addition in the background
        asyncio.create_task(process_graph_addition())

        response = result.get("output", "I'm sorry, I couldn't process your request.")
        
        print(f"Agent response: {response}")
        if not response:
            raise HTTPException(status_code=404, detail="No answer generated")
            
        return ChatResponse(answer=response)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        error_response = "I'm sorry, I encountered an error while processing your request. Please try asking your question differently."
        return ChatResponse(answer=error_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)