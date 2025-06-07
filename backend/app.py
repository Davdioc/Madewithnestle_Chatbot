from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.messages import HumanMessage, SystemMessage

from models import ChatRequest, ChatResponse
from config import CORS_ORIGINS
from agent_setup import init_components
from utils import add_to_graph, get_text_chunks_langchain, graph, agent_executor

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize app
app = FastAPI(title="Made with Nestlé Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize components
llm = init_components()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Made with Nestlé Agent Chatbot API is running!"}

@app.post("/api/chat", response_model=ChatResponse)
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