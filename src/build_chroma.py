#!/usr/bin/env python3
"""
ChromaDB Client for Document Ingestion
Builds an ephemeral ChromaDB client and ingests processed JSON files into collections.
"""

import os
import json
import sys
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except ImportError:
    print("ChromaDB not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb"])
    import chromadb
    from chromadb.config import Settings
    
    
# Grab OpenAI API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 1536


class ChromaDocumentIngester:
    """
    A class to handle ChromaDB document ingestion from processed JSON files.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB client.
        
        Args:
            persist_directory: Directory to persist data (None for ephemeral)
        """
        self.persist_directory = persist_directory
        
        # Persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized persistent ChromaDB client at: {persist_directory}")
        
        self.collections = {}
    
    def get_collection_name(self, paper_title: str) -> str:
        """
        Generate a valid collection name from paper title.
        
        Args:
            paper_title: The title of the paper
            
        Returns:
            Valid collection name
        """
        # Remove special characters and limit length
        # Replace special characters with underscores
        name = re.sub(r'[^a-zA-Z0-9\s]', '_', paper_title)
        
        # Replace multiple spaces/underscores with single underscore
        name = re.sub(r'[\s_]+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Limit length (ChromaDB has limits)
        if len(name) > 63:
            name = name[:63]
        
        # Ensure it starts with a letter
        if name and not name[0].isalpha():
            name = 'paper_' + name
        
        return name.lower()
    
    def create_or_get_collection(self, paper_title: str) -> chromadb.Collection:
        """
        Create or get a collection for a paper.
        
        Args:
            paper_title: The title of the paper
            
        Returns:
            ChromaDB collection
        """
        collection_name = self.get_collection_name(paper_title)
        
        if collection_name in self.collections:
            return self.collections[collection_name]
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Retrieved existing collection: {collection_name}")
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"paper_title": paper_title},
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=EMBEDDING_MODEL,
                    dimensions=DIMENSIONS
                )
            )
            self.logger.info(f"Created new collection: {collection_name}")
        
        self.collections[collection_name] = collection
        return collection
    
    def process_document_chunk(self, chunk: Dict[str, Any], paper_title: str) -> Dict[str, Any]:
        """
        Process a single document chunk for ChromaDB ingestion.
        
        Args:
            chunk: The chunk dictionary from JSON
            paper_title: The title of the paper
            
        Returns:
            Processed chunk for ChromaDB
        """
        chunk_id = str(chunk.get('id', ''))
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Create ChromaDB document
        document = {
            'id': f"{paper_title}_{chunk_id}",
            'document': content,
            'metadata': {
                'paper_title': paper_title,
                'chunk_id': chunk_id,
                'content_type': metadata.get('type', 'unknown'),
                'html_class': metadata.get('html_class', ''),
                'token_count': metadata.get('token_count', 0),
                'position': metadata.get('position', 0),
                'tag_name': metadata.get('tag_name', '')
            }
        }
        
        # Add level for headers
        if metadata.get('type') == 'header' and 'level' in metadata and metadata['level'] is not None:
            document['metadata']['level'] = metadata['level']
        
        # Add list_type for lists
        if metadata.get('type') == 'list' and 'list_type' in metadata and metadata['list_type'] is not None:
            document['metadata']['list_type'] = metadata['list_type']
        
        # Add merged_chunks for tables (ChromaDB compatible)
        if metadata.get('type') == 'table' and 'merged_chunks' in metadata and metadata['merged_chunks'] is not None:
            document['metadata']['merged_chunks'] = metadata['merged_chunks']
        
        return document
    
    def ingest_paper(self, json_file_path: str) -> bool:
        """
        Ingest a single paper JSON file into ChromaDB.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Processing: {json_file_path}")
            
            # Load JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                self.logger.warning(f"No chunks found in {json_file_path}")
                return False
            
            # Get paper title from first chunk
            paper_title = chunks[0].get('metadata', {}).get('name', 'Unknown Paper')
            self.logger.info(f"Paper: {paper_title}")
            
            # If unknown paper, use the file name:
            if paper_title == 'Unknown Paper':
                paper_title = json_file_path.split('/')[-1].split('.')[0]
            
            # Create or get collection
            collection = self.create_or_get_collection(paper_title)
            
            # Process chunks
            documents = []
            ids = []
            metadatas = []
            
            for chunk in chunks:
                processed_chunk = self.process_document_chunk(chunk, paper_title)
                
                documents.append(processed_chunk['document'])
                ids.append(str(processed_chunk['id']))
                metadatas.append(processed_chunk['metadata'])
                            
            # Add to collection
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(chunks)} chunks to collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {json_file_path}: {e}")
            return False
    
    def ingest_folder(self, folder_path: str) -> Dict[str, int]:
        """
        Ingest all JSON files from a folder.
        
        Args:
            folder_path: Path to the folder containing JSON files
            
        Returns:
            Dictionary with processing results
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Folder not found: {folder_path}")
            return {}
        
        # Find all JSON files
        json_files = list(folder.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {folder_path}")
            return {}
        
        print(f"Found {len(json_files)} JSON files to process")
        
        results = {
            'total_files': len(json_files),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0
        }
        
        # Process each file
        for json_file in json_files:
            success = self.ingest_paper(str(json_file))
            
            if success:
                results['successful'] += 1
                # Count chunks in successful files
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    results['total_chunks'] += len(chunks)
                except:
                    pass
            else:
                results['failed'] += 1
        
        return results
    
    def list_collections(self) -> List[str]:
        """
        List all collection names in ChromaDB.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            return {
                'name': collection_name,
                'count': count,
                'metadata': collection.metadata
            }
        except Exception as e:
            print(f"Error getting collection info for {collection_name}: {e}")
            return {'name': collection_name, 'count': 0, 'metadata': {}}
    
    def search_collection(self, collection_name: str, query: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Search a collection for documents.
        
        Args:
            collection_name: Name of the collection to search
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results or None if error
        """
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
            return None
    
    def delete_all_collections(self):
        """
        Delete all collections from ChromaDB.
        """
        for collection_name in self.collections:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")




def main():
    """Main function to run the document ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest processed JSON files into ChromaDB')
    parser.add_argument('--folder', default='documents/processed/database', 
                       help='Folder containing JSON files (default: documents/processed/database)')
    parser.add_argument('--persist', help='Directory to persist ChromaDB data (ephemeral if not specified)')
    parser.add_argument('--list-collections', action='store_true', 
                       help='List all collections after ingestion')
    parser.add_argument('--search', help='Search query to test after ingestion')
    parser.add_argument('--collection', help='Collection name for search')
    
    args = parser.parse_args()
    
    # Initialize client
    ingester = ChromaDocumentIngester(persist_directory=args.persist)
    
    # Ingest documents
    print("=== ChromaDB Document Ingestion ===\n")
    results = ingester.ingest_folder(args.folder)
    
    # Print results
    print(f"\n=== Ingestion Results ===")
    print(f"Total files: {results.get('total_files', 0)}")
    print(f"Successful: {results.get('successful', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Total chunks ingested: {results.get('total_chunks', 0)}")
    
    # List collections if requested
    if args.list_collections:
        print(f"\n=== Collections ===")
        collections = ingester.list_collections()
        for collection_name in collections:
            info = ingester.get_collection_info(collection_name)
            print(f"  {collection_name}: {info.get('count', 0)} documents")
    
    # Test search if requested
    if args.search and args.collection:
        print(f"\n=== Search Test ===")
        print(f"Query: {args.search}")
        print(f"Collection: {args.collection}")
        
        results = ingester.search_collection(args.collection, args.search)
        if results and 'documents' in results:
            for i, doc in enumerate(results['documents'][0]):
                print(f"  Result {i+1}: {doc[:100]}...")
    
    print("\n=== Ingestion Complete ===")


if __name__ == "__main__":
    main()
