import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from models import ChatRequest, ChatResponse, AddNode, AddNodeResponse
from graph_service import graph, get_text_chunks_langchain, add_to_graph
from agent_service import agent_executor

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

router = APIRouter()

@router.get("/")
def read_root():
    return {"status": "ok", "message": "Made with Nestlé Agent Chatbot API is running."}

@router.post("/api/add", response_model=AddNodeResponse)
async def add_to_graphDB(request: AddNode):
    try:
        from agent_service import init_llm
        
        if not graph:
            raise HTTPException(status_code=500, detail="Graph database not initialized")
        
        llm = init_llm()
        
        # Split text into chunks
        text_chunks = get_text_chunks_langchain(request.text)
        
        # Convert to graph documents
        llm_transformer = LLMGraphTransformer(llm=llm)
        add_to_graph(text_chunks, graph, llm_transformer)
        
        return AddNodeResponse(status="success")
        
    except Exception as e:
        print(f"Error adding node: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add node to graph")

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
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
        result = await run_agent()

        response = result.get("output", "I'm sorry, I couldn't process your request.")
        
        print(f"Agent response: {response}")
        if not response:
            raise HTTPException(status_code=404, detail="No answer generated")
            
        return ChatResponse(answer=response)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        error_response = "I'm sorry, I encountered an error while processing your request. Please try asking your question differently."
        return ChatResponse(answer=error_response)