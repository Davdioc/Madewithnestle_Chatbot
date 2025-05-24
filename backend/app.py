from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import re
from dotenv import load_dotenv

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from neo4j import GraphDatabase

#load environment variables from .env file
load_dotenv()

#Initialize app
app = FastAPI(title="Made with Nestlé Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class Entities(BaseModel):
    """Identifying information about entities."""
    link: List[str] = Field(
        ...,
        description="What is being searched for.",
    )

def init_components():
    #Initialize Neo4j Graph DB connection
    graph = Neo4jGraph()
    
    #Initialize Azure OpenAI components
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )
    
    # Initialize LLM Graph Transformer
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_type="azure"
    )
    
    # Initialize vector store
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_retriever = vector_index.as_retriever()

    parser = PydanticOutputParser(pydantic_object=Entities)
    
    #Create an entity extraction prompt
    entity_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant for the MadeWithNestlé website. Your role is to assist users with general inquiries by providing clear, relevant, and accurate information. Always include a reference link to the original source of the information when responding.\n"
            "{format_instructions}"
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ])
    
    #create entity chain
    entity_chain = entity_prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

    template = """Answer the question based only on the following context and add a url of the source of the information if any towards the end of the statement. Never make up a url: {context}
    
    Question: {question}
    Use natural language and be friendly. Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": lambda question: full_retriever(question, graph, vector_retriever, entity_chain),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return graph, vector_retriever, entity_chain, chain, llm_transformer

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

def add_to_graph(documents, graph, llm_transformer):
    """Add documents to the graph database"""
    if not documents:
        return
        
    graph_documents = []
    for doc in documents:
        graph_doc = llm_transformer.convert_to_graph_documents([doc])
        graph_documents.extend(graph_doc)
    print(f"Adding {len(graph_documents)} documents to the graph")
    print("Graph documents:", graph_documents)
    if graph_documents:
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

def remove_lucene_chars(text):
    """
    Properly escape or remove special Lucene characters from the input string
    to prevent query parsing errors.
    """
    special_chars = r'+-&|!(){}[]^"~*?:\/'
    
    #Replace or escape each special character
    cleaned_text = ""
    for char in text:
        if char in special_chars:
            cleaned_text += " "  # Replace with space
        else:
            cleaned_text += char
    
    # Remove extra spaces and trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def generate_full_text_query(input_text: str) -> str:
    words = [el for el in remove_lucene_chars(input_text).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()

def graph_retriever(question: str, graph: Neo4jGraph, entity_chain) -> str:
    #Collects the neighborhood of entities mentioned in the question
    result = ""
    try:
        entities = entity_chain.invoke(question)
        
        for entity in entities.link:
            #Prepare the entity before using in query
            sanitized_entity = remove_lucene_chars(entity)
            
            #skip empty queries
            if not sanitized_entity:
                continue
            
            try:
                response = graph.query(
                    """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:7})
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
                    RETURN output LIMIT 50
                    """,
                    {"query": sanitized_entity},
                )
                result += "\n".join([el['output'] for el in response])
            except Exception as e:
                print(f"Error querying graph for entity '{entity}': {str(e)}")
                #Carry on with next entity rather than failing completely
                continue
    
    except Exception as e:
        print(f"Error in entity extraction: {str(e)}")
        #return empty result rather than crashing
        pass
    
    return result

#function to combine graph data and vector data
def full_retriever(question: str, graph: Neo4jGraph, vector_retriever, entity_chain):
    graph_data = graph_retriever(question, graph, entity_chain)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f"""Graph data:
    {graph_data}\n\nVector data:
    {"". join(vector_data)}
    """
    return final_data

#Initialize components
graph, vector_retriever, entity_chain, chain, llm_transformer = init_components()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Made with Nestlé Chatbot API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = chain.invoke(input= request.question)
        print("LLM response:", response)
        
        documents = [request.question] if request.question.strip() else []
        
        # Add to graph only if documents is not empty
        if documents:
            documents = get_text_chunks_langchain(request.question)
            add_to_graph(documents, graph, llm_transformer)
        
        if not response:
            raise HTTPException(status_code=404, detail="No answer found")
        return ChatResponse(answer=response)
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)