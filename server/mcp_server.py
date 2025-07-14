from typing import Dict, List
import chromadb
from mcp.server.fastmcp import FastMCP
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# Setup logging
def setup_logging():
    """Setup logging configuration for the MCP server."""
    # Create server directory if it doesn't exist
    server_dir = Path(__file__).parent
    server_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(server_dir / 'mcp_server.log'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

##########################################################################
####################### Instantiate MCP Server ###########################
##########################################################################

mcp = FastMCP("chroma")

##########################################################################
####################### Global Variables #################################
##########################################################################

_chroma_client = None
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 1536


def get_chroma_client():
    """
    Get the ChromaDB client.
    """
    global _chroma_client
    if _chroma_client is None:
        logger.info("Initializing ChromaDB persistent client")
        try:
            _chroma_client = chromadb.PersistentClient(path="./chroma_db")
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    return _chroma_client


##########################################################################
####################### Collection Tools #################################
##########################################################################

@mcp.tool()
async def test_tool(message: str = "Hello, world!") -> str:
    """
    Test tool to ensure the MCP server is working.
    """
    logger.info(f"Test tool called with message: {message}")
    result = f"Test tool received message: {message}"
    logger.info(f"Test tool returning: {result}")
    return result


@mcp.tool()
async def chroma_list_collections(
    limit: int | None = None,
    offset: int | None = None
) -> List[str]:
    """List all collection names in the Chroma database with pagination support.
    
    Args:
        limit: Optional maximum number of collections to return
        offset: Optional number of collections to skip before returning results
    
    Returns:
        List of collection names or ["__NO_COLLECTIONS_FOUND__"] if database is empty
    """
    logger.info(f"Listing collections with limit={limit}, offset={offset}")
    client = get_chroma_client()
    try:
        colls = client.list_collections(limit=limit, offset=offset)
        # Safe handling: If colls is None or empty, return a special marker
        if not colls:
            logger.info("No collections found in database")
            return ["__NO_COLLECTIONS_FOUND__"]
        # Otherwise iterate to get collection names
        collection_names = [coll.name for coll in colls]
        logger.info(f"Found {len(collection_names)} collections: {collection_names}")
        return collection_names

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise Exception(f"Failed to list collections: {str(e)}") from e

@mcp.tool()
async def chroma_get_collection_info(collection_name: str) -> Dict:
    """Get information about a Chroma collection.
    
    Args:
        collection_name: Name of the collection to get info about
    """
    logger.info(f"Getting collection info for '{collection_name}'")
    client = get_chroma_client()
    try:
        collection = client.get_collection(
            collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMBEDDING_MODEL,
                dimensions=DIMENSIONS
            )
        )
        logger.info(f"Retrieved collection '{collection_name}'")
        
        # Get collection count
        count = collection.count()
        logger.info(f"Collection '{collection_name}' has {count} documents")
        
        # Peek at a few documents
        peek_results = collection.peek(limit=1)
        logger.info(f"Retrieved {len(peek_results.get('ids', []))} sample documents")
        
        result = {
            "name": collection_name,
            "paper_title": peek_results['metadatas'][0]['paper_title'],
            "count": count,
            "sample_documents": peek_results
        }
        logger.info(f"Collection info retrieved successfully for '{collection_name}'")
        return result
    except Exception as e:
        logger.error(f"Failed to get collection info for '{collection_name}': {e}")
        raise Exception(f"Failed to get collection info for '{collection_name}': {str(e)}") from e
    
@mcp.tool()
async def chroma_query_documents(
    collection_name: str,
    query_texts: List[str],
    n_results: int = 5,
    where: Dict | None = None,
    where_document: Dict | None = None,
    include: List[str] = ["documents", "metadatas", "distances"]
) -> Dict:
    """Query documents from a Chroma collection with advanced filtering.
    
    Args:
        collection_name: Name of the collection to query
        query_texts: List of query texts to search for
        n_results: Number of results to return per query
        where: Optional metadata filters using Chroma's query operators
               Examples:
               - Simple equality: {"metadata_field": "value"}
               - Comparison: {"metadata_field": {"$gt": 5}}
               - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: Optional document content filters
        include: List of what to include in response. By default, this will include documents, metadatas, and distances.
    """
    logger.info(f"Querying collection '{collection_name}' with {len(query_texts)} queries, n_results={n_results}")
    logger.debug(f"Query texts: {query_texts}")
    logger.debug(f"Where filter: {where}")
    logger.debug(f"Where document filter: {where_document}")
    logger.debug(f"Include: {include}")
    
    if not query_texts:
        logger.error("Query texts list is empty")
        raise ValueError("The 'query_texts' list cannot be empty.")

    client = get_chroma_client()
    try:
        collection = client.get_collection(
            collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMBEDDING_MODEL,
                dimensions=DIMENSIONS
            )
        )
        logger.info(f"Retrieved collection '{collection_name}' for querying")
        
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        # Log query results summary
        total_results = len(results.get('ids', [])) if results.get('ids') else 0
        logger.info(f"Query completed successfully. Found {total_results} results for collection '{collection_name}'")
        
        return results
    except Exception as e:
        logger.error(f"Failed to query documents from collection '{collection_name}': {e}")
        raise Exception(f"Failed to query documents from collection '{collection_name}': {str(e)}") from e
    
@mcp.tool()
async def chroma_get_documents(
    collection_name: str,
    ids: List[str] | None = None,
    where: Dict | None = None,
    where_document: Dict | None = None,
    include: List[str] = ["documents", "metadatas"],
    limit: int | None = None,
    offset: int | None = None
) -> Dict:
    """Get documents from a Chroma collection with optional filtering.
    
    Args:
        collection_name: Name of the collection to get documents from
        ids: Optional list of document IDs to retrieve
        where: Optional metadata filters using Chroma's query operators
               Examples:
               - Simple equality: {"metadata_field": "value"}
               - Comparison: {"metadata_field": {"$gt": 5}}
               - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: Optional document content filters
        include: List of what to include in response. By default, this will include documents, and metadatas.
        limit: Optional maximum number of documents to return
        offset: Optional number of documents to skip before returning results
    
    Returns:
        Dictionary containing the matching documents, their IDs, and requested includes
    """
    logger.info(f"Getting documents from collection '{collection_name}'")
    logger.debug(f"IDs: {ids}")
    logger.debug(f"Where filter: {where}")
    logger.debug(f"Where document filter: {where_document}")
    logger.debug(f"Include: {include}")
    logger.debug(f"Limit: {limit}, Offset: {offset}")
    
    client = get_chroma_client()
    try:
        collection = client.get_collection(
            collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMBEDDING_MODEL,
                dimensions=DIMENSIONS
            )
        )
        logger.info(f"Retrieved collection '{collection_name}' for document retrieval")
        
        results = collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset
        )
        
        # Log results summary
        total_documents = len(results.get('ids', [])) if results.get('ids') else 0
        logger.info(f"Retrieved {total_documents} documents from collection '{collection_name}'")
        
        return results
    except Exception as e:
        logger.error(f"Failed to get documents from collection '{collection_name}': {e}")
        raise Exception(f"Failed to get documents from collection '{collection_name}': {str(e)}") from e
    
        
def main():
    """Entry point for the server"""
    logger.info("Starting MCP server")
    
    try:
        logger.info("Initializing ChromaDB client...")
        get_chroma_client()
        logger.info("ChromaDB client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {str(e)}")
        print(f"Error initializing ChromaDB client: {str(e)}")
        return
    
    logger.info("Starting MCP server with stdio transport")
    # Run the MCP server
    mcp.run(transport="stdio")
    
if __name__ == "__main__":
    main()
    
    