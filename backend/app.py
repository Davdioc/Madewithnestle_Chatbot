from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from routes import router
from graph_service import init_graph_components
from agent_service import init_agent

# Initialize app
app = FastAPI(title="Made with Nestlé Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    """Initialize all components on application startup"""
    try:
        # Initialize graph components
        init_graph_components()
        
        # Initialize agent
        init_agent()
        
        print("All components initialized successfully!")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)