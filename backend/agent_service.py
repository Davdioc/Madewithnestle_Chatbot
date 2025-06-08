from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import search_nearby_locations, get_amazon_product_links, search_nestle_knowledge_base
from config import Config

# Global variables
agent_executor = None

def init_llm():
    """Initialize Azure OpenAI LLM"""
    llm = AzureChatOpenAI(
        deployment_name=Config.AZURE_OPENAI_DEPLOYMENT,
        openai_api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        openai_api_version=Config.AZURE_OPENAI_API_VERSION,
        temperature=0.3
    )
    return llm

def init_agent():
    """Initialize the agent with tools and prompt"""
    global agent_executor
    
    llm = init_llm()
    
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