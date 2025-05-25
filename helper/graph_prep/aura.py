import pickle
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain.chat_models import AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Neo4jDataLoader:
    def __init__(self):
        """Initialize the Neo4j connection and LLM components"""
        self.neo4j_uri = os.getenv("NEO4J_URI") 
        self.neo4j_username =  os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD") 
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )

        logger.info("Neo4j connection and LLM components initialized successfully")

    def test_connection(self):
        try:
            result = self.graph.query("RETURN 'Connection successful' as message")
            logger.info(f"Neo4j connection test: {result[0]['message']}")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {str(e)}")
            return False

    def clear_existing_data(self):
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            logger.info("Existing data cleared from Neo4j")
        except Exception as e:
            logger.error(f"Error clearing existing data: {str(e)}")

    def load_graph_documents_from_pickle(self, pickle_file_path):
        try:
            with open(pickle_file_path, 'rb') as f:
                loaded_graph_documents = pickle.load(f)
            logger.info(f"Successfully loaded {len(loaded_graph_documents)} graph documents from pickle file")
            return loaded_graph_documents
        except Exception as e:
            logger.error(f"Error loading pickle file: {str(e)}")
            return None

    def add_graph_documents_to_neo4j(self, graph_documents):
        try:
            if not graph_documents:
                logger.warning("No graph documents to add")
                return False

            logger.info(f"Adding {len(graph_documents)} graph documents to Neo4j...")
            
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            
            logger.info("Successfully added graph documents to Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Error adding graph documents to Neo4j: {str(e)}")
            return False

    def create_indexes(self):
        try:
            index_queries = [
                "CREATE FULLTEXT INDEX fulltext_entity_id IF NOT EXISTS FOR (n) ON EACH [n.id]",
                "CREATE INDEX entity_id_index IF NOT EXISTS FOR (n) ON (n.id)",
                "CREATE INDEX document_text_index IF NOT EXISTS FOR (n:Document) ON (n.text)"
            ]
            
            for query in index_queries:
                try:
                    self.graph.query(query)
                    logger.info(f"Created index: {query}")
                except Exception as e:
                    logger.warning(f"Index creation warning (may already exist): {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")

    def get_database_stats(self):
        #Get statistics about the data in Neo4j
        try:
            node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
            labels = self.graph.query("CALL db.labels() YIELD label RETURN collect(label) as labels")[0]['labels']
            rel_types = self.graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")[0]['types']
            
            stats = {
                'nodes': node_count,
                'relationships': rel_count,
                'node_labels': labels,
                'relationship_types': rel_types
            }
            
            logger.info(f"Database Stats - Nodes: {node_count}, Relationships: {rel_count}")
            logger.info(f"Node Labels: {labels}")
            logger.info(f"Relationship Types: {rel_types}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return None

def main():
    loader = Neo4jDataLoader()
    
    #test Neo4j connection
    if not loader.test_connection():
        logger.error("Cannot proceed without valid Neo4j connection")
        return
    loader.clear_existing_data()
    pickle_file_path = "graph_documents.pkl"  # Update this path
    if os.path.exists(pickle_file_path):
        logger.info("Loading graph documents from pickle file...")
        print("Loading graph documents from pickle file...")
        graph_documents = loader.load_graph_documents_from_pickle(pickle_file_path)
        
        if graph_documents:
            success = loader.add_graph_documents_to_neo4j(graph_documents)
            if success:
                logger.info("Graph documents successfully loaded from pickle file")
    logger.info("Data loading process completed")

if __name__ == "__main__":
    main()