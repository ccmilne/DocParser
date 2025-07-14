#!/usr/bin/env python3
"""
Document Processing Pipeline Orchestrator

This script orchestrates the entire workflow for processing PDF documents:
1. PDF to HTML conversion using Gemini
2. HTML to JSON parsing with content type classification
3. JSON to database chunks conversion
4. ChromaDB vector database creation

The pipeline only processes files that don't already have corresponding output files,
ensuring efficient incremental processing.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline components
from src.doc_parser import generate_html_from_pdf
from src.html_parser import parse_html_content, save_chunks_to_json
from src.chunker import convert_json_to_database_format
from src.build_chroma import ChromaDocumentIngester


class DocumentProcessingPipeline:
    """
    Orchestrates the complete document processing pipeline.
    """
    
    def __init__(self, 
                 pdf_folder: str = "documents/pdfs",
                 html_folder: str = "documents/processed/HTML",
                 json_folder: str = "documents/processed/JSON",
                 database_folder: str = "documents/processed/database",
                 chroma_persist_dir: str = "./chroma_db"):
        """
        Initialize the processing pipeline.
        
        Args:
            pdf_folder: Directory containing PDF files
            html_folder: Directory for generated HTML files
            json_folder: Directory for parsed JSON files
            database_folder: Directory for database-ready chunks
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        self.pdf_folder = Path(pdf_folder)
        self.html_folder = Path(html_folder)
        self.json_folder = Path(json_folder)
        self.database_folder = Path(database_folder)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        
        # Ensure all directories exist
        self.pdf_folder.mkdir(parents=True, exist_ok=True)
        self.html_folder.mkdir(parents=True, exist_ok=True)
        self.json_folder.mkdir(parents=True, exist_ok=True)
        self.database_folder.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize ChromaDB ingester
        self.chroma_ingester = ChromaDocumentIngester(persist_directory=str(self.chroma_persist_dir))
        
        # Check API keys
        self._check_api_keys()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('src/run_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _check_api_keys(self):
        """Check that required API keys are available."""
        if not os.getenv("GEMINI_API_KEY"):
            self.logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY is required for PDF processing")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.logger.warning("OPENAI_API_KEY not found - agent interface may not work")
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the PDF folder."""
        if not self.pdf_folder.exists():
            self.logger.warning(f"PDF folder does not exist: {self.pdf_folder}")
            return []
        
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def get_file_basename(self, file_path: Path) -> str:
        """Get the base name of a file (without extension)."""
        return file_path.stem
    
    def needs_html_processing(self, pdf_path: Path) -> bool:
        """Check if a PDF needs HTML processing."""
        pdf_basename = self.get_file_basename(pdf_path)
        html_path = self.html_folder / f"{pdf_basename}.html"
        
        if not html_path.exists():
            self.logger.info(f"PDF needs HTML processing: {pdf_path.name}")
            return True
        
        # Check if PDF is newer than HTML
        if pdf_path.stat().st_mtime > html_path.stat().st_mtime:
            self.logger.info(f"PDF is newer than HTML: {pdf_path.name}")
            return True
        
        self.logger.info(f"HTML already exists and is current: {pdf_path.name}")
        return False
    
    def needs_json_processing(self, pdf_path: Path) -> bool:
        """Check if a PDF needs JSON processing."""
        pdf_basename = self.get_file_basename(pdf_path)
        html_path = self.html_folder / f"{pdf_basename}.html"
        json_path = self.json_folder / f"{pdf_basename}_chunks.json"
        
        if not html_path.exists():
            self.logger.info(f"No HTML file for JSON processing: {pdf_path.name}")
            return False
        
        if not json_path.exists():
            self.logger.info(f"PDF needs JSON processing: {pdf_path.name}")
            return True
        
        # Check if HTML is newer than JSON
        if html_path.stat().st_mtime > json_path.stat().st_mtime:
            self.logger.info(f"HTML is newer than JSON: {pdf_path.name}")
            return True
        
        self.logger.info(f"JSON already exists and is current: {pdf_path.name}")
        return False
    
    def needs_database_processing(self, pdf_path: Path) -> bool:
        """Check if a PDF needs database processing."""
        pdf_basename = self.get_file_basename(pdf_path)
        json_path = self.json_folder / f"{pdf_basename}_chunks.json"
        database_path = self.database_folder / f"{pdf_basename}_database.json"
        
        if not json_path.exists():
            self.logger.info(f"No JSON file for database processing: {pdf_path.name}")
            return False
        
        if not database_path.exists():
            self.logger.info(f"PDF needs database processing: {pdf_path.name}")
            return True
        
        # Check if JSON is newer than database
        if json_path.stat().st_mtime > database_path.stat().st_mtime:
            self.logger.info(f"JSON is newer than database: {pdf_path.name}")
            return True
        
        self.logger.info(f"Database already exists and is current: {pdf_path.name}")
        return False
    
    def needs_chroma_processing(self, pdf_path: Path) -> bool:
        """Check if a PDF needs ChromaDB processing."""
        pdf_basename = self.get_file_basename(pdf_path)
        database_path = self.database_folder / f"{pdf_basename}_database.json"
        
        if not database_path.exists():
            self.logger.info(f"No database file for ChromaDB processing: {pdf_path.name}")
            return False
        
        # For now, always process ChromaDB since we don't have a reliable way to check
        # if the data is already in ChromaDB without querying it
        self.logger.info(f"PDF needs ChromaDB processing: {pdf_path.name}")
        return True
    
    def process_pdf_to_html(self, pdf_path: Path) -> bool:
        """Convert PDF to HTML using Gemini AI."""
        try:
            self.logger.info(f"Converting PDF to HTML: {pdf_path.name}")
            generate_html_from_pdf(str(pdf_path))
            self.logger.info(f"Successfully converted {pdf_path.name} to HTML")
            return True
        except Exception as e:
            self.logger.error(f"Failed to convert {pdf_path.name} to HTML: {e}")
            return False
    
    def process_html_to_json(self, pdf_path: Path) -> bool:
        """Convert HTML to JSON with content type classification."""
        try:
            pdf_basename = self.get_file_basename(pdf_path)
            html_path = self.html_folder / f"{pdf_basename}.html"
            json_path = self.json_folder / f"{pdf_basename}_chunks.json"
            
            self.logger.info(f"Converting HTML to JSON: {pdf_path.name}")
            
            # Read HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML to chunks
            chunks = parse_html_content(html_content)
            
            # Save to JSON
            save_chunks_to_json(chunks, str(json_path))
            
            self.logger.info(f"Successfully converted {pdf_path.name} to JSON")
            return True
        except Exception as e:
            self.logger.error(f"Failed to convert {pdf_path.name} to JSON: {e}")
            return False
    
    def process_json_to_database(self, pdf_path: Path) -> bool:
        """Convert JSON to database-ready chunks."""
        try:
            pdf_basename = self.get_file_basename(pdf_path)
            json_path = self.json_folder / f"{pdf_basename}_chunks.json"
            database_path = self.database_folder / f"{pdf_basename}_database.json"
            
            self.logger.info(f"Converting JSON to database: {pdf_path.name}")
            
            # Convert JSON to database format
            convert_json_to_database_format(str(json_path), str(database_path))
            
            self.logger.info(f"Successfully converted {pdf_path.name} to database format")
            return True
        except Exception as e:
            self.logger.error(f"Failed to convert {pdf_path.name} to database format: {e}")
            return False
    
    def process_database_to_chroma(self, pdf_path: Path) -> bool:
        """Ingest database file into ChromaDB."""
        try:
            pdf_basename = self.get_file_basename(pdf_path)
            database_path = self.database_folder / f"{pdf_basename}_database.json"
            
            self.logger.info(f"Ingesting into ChromaDB: {pdf_path.name}")
            
            # Ingest into ChromaDB
            success = self.chroma_ingester.ingest_paper(str(database_path))
            
            if success:
                self.logger.info(f"Successfully ingested {pdf_path.name} into ChromaDB")
            else:
                self.logger.warning(f"Failed to ingest {pdf_path.name} into ChromaDB")
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to ingest {pdf_path.name} into ChromaDB: {e}")
            return False
    
    def process_single_pdf(self, pdf_path: Path) -> Dict[str, bool]:
        """Process a single PDF through the entire pipeline."""
        self.logger.info(f"Processing PDF: {pdf_path.name}")
        
        results = {
            'pdf_to_html': False,
            'html_to_json': False,
            'json_to_database': False,
            'database_to_chroma': False
        }
        
        # Stage 1: PDF to HTML
        if self.needs_html_processing(pdf_path):
            results['pdf_to_html'] = self.process_pdf_to_html(pdf_path)
        else:
            results['pdf_to_html'] = True  # Already processed
        
        # Stage 2: HTML to JSON
        if results['pdf_to_html'] and self.needs_json_processing(pdf_path):
            results['html_to_json'] = self.process_html_to_json(pdf_path)
        else:
            results['html_to_json'] = True  # Already processed
        
        # Stage 3: JSON to Database
        if results['html_to_json'] and self.needs_database_processing(pdf_path):
            results['json_to_database'] = self.process_json_to_database(pdf_path)
        else:
            results['json_to_database'] = True  # Already processed
        
        # Stage 4: Database to ChromaDB
        if results['json_to_database'] and self.needs_chroma_processing(pdf_path):
            results['database_to_chroma'] = self.process_database_to_chroma(pdf_path)
        else:
            results['database_to_chroma'] = True  # Already processed or not needed
        
        return results
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete processing pipeline for all PDFs."""
        self.logger.info("Starting document processing pipeline")
        
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            self.logger.warning("No PDF files found to process")
            return {'status': 'no_files', 'processed': []}
        
        results = {
            'status': 'completed',
            'total_pdfs': len(pdf_files),
            'processed': [],
            'errors': []
        }
        
        for pdf_path in pdf_files:
            try:
                pdf_result = self.process_single_pdf(pdf_path)
                results['processed'].append({
                    'pdf_name': pdf_path.name,
                    'results': pdf_result,
                    'success': all(pdf_result.values())
                })
                
                if not all(pdf_result.values()):
                    results['errors'].append({
                        'pdf_name': pdf_path.name,
                        'failed_stages': [k for k, v in pdf_result.items() if not v]
                    })
                    
            except Exception as e:
                self.logger.error(f"Unexpected error processing {pdf_path.name}: {e}")
                results['errors'].append({
                    'pdf_name': pdf_path.name,
                    'error': str(e)
                })
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of the processing results."""
        self.logger.info("=" * 50)
        self.logger.info("PROCESSING PIPELINE SUMMARY")
        self.logger.info("=" * 50)
        
        total_pdfs = results['total_pdfs']
        successful = sum(1 for p in results['processed'] if p['success'])
        failed = total_pdfs - successful
        
        self.logger.info(f"Total PDFs: {total_pdfs}")
        self.logger.info(f"Successfully processed: {successful}")
        self.logger.info(f"Failed: {failed}")
        
        if results['errors']:
            self.logger.info("\nErrors:")
            for error in results['errors']:
                self.logger.info(f"  - {error['pdf_name']}: {error.get('failed_stages', error.get('error', 'Unknown error'))}")
        
        # ChromaDB status
        collections = self.chroma_ingester.list_collections()
        self.logger.info(f"\nChromaDB Collections: {len(collections)}")
        for collection in collections:
            info = self.chroma_ingester.get_collection_info(collection)
            self.logger.info(f"  - {collection}: {info['count']} documents")
        
        self.logger.info("=" * 50)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get the current processing status of all PDFs."""
        pdf_files = self.get_pdf_files()
        status = {
            'total_pdfs': len(pdf_files),
            'pdfs': []
        }
        
        for pdf_path in pdf_files:
            pdf_basename = self.get_file_basename(pdf_path)
            pdf_status = {
                'name': pdf_path.name,
                'html_exists': (self.html_folder / f"{pdf_basename}.html").exists(),
                'json_exists': (self.json_folder / f"{pdf_basename}_chunks.json").exists(),
                'database_exists': (self.database_folder / f"{pdf_basename}_database.json").exists(),
                'needs_html': self.needs_html_processing(pdf_path),
                'needs_json': self.needs_json_processing(pdf_path),
                'needs_database': self.needs_database_processing(pdf_path),
                'needs_chroma': self.needs_chroma_processing(pdf_path)
            }
            status['pdfs'].append(pdf_status)
        
        return status


def main():
    """Main entry point for the processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    parser.add_argument("--pdf-folder", default="documents/pdfs", help="PDF folder path")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB persistence directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = DocumentProcessingPipeline(
            pdf_folder=args.pdf_folder,
            chroma_persist_dir=args.chroma_dir
        )
        
        if args.status:
            # Show status only
            status = pipeline.get_processing_status()
            print(json.dumps(status, indent=2))
        else:
            # Run full pipeline
            results = pipeline.run_pipeline()
            print(f"\nPipeline completed with status: {results['status']}")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
