import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.graphs import Neo4jGraph

# Cache for compiled regex patterns
_lucene_chars_pattern = re.compile(r'[+\-&|!(){}[\]^"~*?:\\/]')
_whitespace_pattern = re.compile(r'\s+')
_text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Global variables for components
graph = None
vector_retriever = None
agent_executor = None

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

def get_text_chunks_langchain(text):
    """Optimized text chunking using cached splitter"""
    docs = [Document(page_content=x) for x in _text_splitter.split_text(text)]
    return docs