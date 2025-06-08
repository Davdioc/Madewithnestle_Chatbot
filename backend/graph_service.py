import re
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from tenacity import retry, stop_after_attempt, wait_fixed
from config import Config

# Cache for compiled regex patterns
_lucene_chars_pattern = re.compile(r'[+\-&|!(){}[\]^"~*?:\\/]')
_whitespace_pattern = re.compile(r'\s+')
_text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Global variables for components
graph = None
vector_retriever = None

def remove_lucene_chars(text):
    """Clean text for Lucene search"""
    cleaned_text = _lucene_chars_pattern.sub(' ', text)
    cleaned_text = _whitespace_pattern.sub(' ', cleaned_text).strip()
    return cleaned_text

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def safe_graph_query(graph, query, params):
    return graph.query(query, params)

def graph_retriever_simple(question: str, graph: Neo4jGraph) -> str:
    """Simplified graph retriever without entity extraction"""
    try:
        # Extract key terms from the question
        key_terms = [term.strip() for term in question.lower().split() 
                    if len(term.strip()) > 3 and term.strip() not in ['what', 'where', 'when', 'why', 'how', 'the', 'and', 'for', 'with']]
        
        if not key_terms:
            return ""
            
        results = []
        for term in key_terms[:5]:  # Limit to first 5 key terms
            try:
                sanitized_term = remove_lucene_chars(term)
                if sanitized_term:
                    outgoing_query = """
                    CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:4})
                    YIELD node
                    MATCH (node)-[r:!MENTIONS]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    LIMIT 25
                    """
                    response1 = safe_graph_query(graph, outgoing_query, {"query": sanitized_term})
                    
                    # Incoming relations
                    incoming_query = """
                    CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:4})
                    YIELD node
                    MATCH (node)<-[r:!MENTIONS]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                    LIMIT 25
                    """
                    response2 = safe_graph_query(graph, incoming_query, {"query": sanitized_term})

                    results.extend([el['output'] for el in response1] + [el['output'] for el in response2])
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

def get_text_chunks_langchain(text):
    """Optimized text chunking using cached splitter"""
    docs = [Document(page_content=x) for x in _text_splitter.split_text(text)]
    return docs

def init_graph_components():
    """Initialize graph database and vector components"""
    global graph, vector_retriever
    
    # Initialize Neo4j Graph DB connection
    graph = Neo4jGraph(
        url=Config.NEO4J_URI,
        username=Config.NEO4J_USERNAME,
        password=Config.NEO4J_PASSWORD,
    )
    print("Connected to Neo4j graph database")
    
    # Initialize embeddings and vector store
    embeddings = AzureOpenAIEmbeddings(
        model=Config.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        openai_api_key=Config.AZURE_OPENAI_EMBEDDINGS_API,
        azure_endpoint=Config.AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
        openai_api_version=Config.AZURE_OPENAI_API_VERSION,
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
    
    return graph, vector_retriever