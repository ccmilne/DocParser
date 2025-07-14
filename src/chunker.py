#!/usr/bin/env python3
"""
JSON to Database Converter
Converts parsed HTML chunks from JSON format into database-ready dictionaries.
"""

import json
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
import re

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.html_parser import ContentType


def count_tokens(text: str) -> int:
    """
    Count tokens in text using a simple word-based approach.
    This is a basic implementation - you might want to use a more sophisticated tokenizer.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Number of tokens
    """
    if not text or not text.strip():
        return 0
    
    # Simple tokenization: split on whitespace and punctuation
    # Remove HTML tags and special characters
    clean_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Replace punctuation with spaces
    
    # Split on whitespace and filter out empty strings
    tokens = [token for token in clean_text.split() if token.strip()]
    
    return len(tokens)


def extract_authors_and_institutions(chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[str], str]:
    """
    Extract authors and their institutions from the parsed chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Tuple of (authors_list, institutions_list, authors_text)
    """
    authors = []
    institutions = []
    authors_text = ""
    
    # Common patterns for author sections
    author_patterns = [
        r'class="authors?"',
        r'class="author"',
        r'class="byline"',
        r'class="contributors"',
        r'id="authors?"',
        r'id="author"'
    ]
    
    # Look for author sections by class/id attributes
    for chunk in chunks:
        attributes = chunk.get('attributes', {})
        class_attr = attributes.get('class', '')
        id_attr = attributes.get('id', '')
        
        # Convert attributes to strings if they're not already
        if isinstance(class_attr, list):
            class_attr = ' '.join(class_attr)
        elif not isinstance(class_attr, str):
            class_attr = str(class_attr)
            
        if isinstance(id_attr, list):
            id_attr = ' '.join(id_attr)
        elif not isinstance(id_attr, str):
            id_attr = str(id_attr)
        
        # Check if this chunk is likely an author section
        is_author_section = False
        for pattern in author_patterns:
            if re.search(pattern, class_attr, re.IGNORECASE) or re.search(pattern, id_attr, re.IGNORECASE):
                is_author_section = True
                break
        
        if is_author_section or 'author' in class_attr.lower():
            content = chunk.get('content', '')
            if content:
                authors_text = content
                authors, institutions = parse_authors_and_institutions(content)
                break
    
    # If no author section found by attributes, look for common patterns in content
    if not authors_text:
        for chunk in chunks:
            content = chunk.get('content', '')
            if not content:
                continue
            
            # Look for patterns that indicate author information
            if any(pattern in content.lower() for pattern in [
                'cameron milne', 'yezzi angi lee', 'taylor wilson', 'hector ferronato',
                'reveal global consulting', 'census bureau', 'fulton, md'
            ]):
                authors_text = content
                authors, institutions = parse_authors_and_institutions(content)
                break
            
            # Look for patterns with commas and institutions
            if ',' in content and any(word in content.lower() for word in ['consulting', 'bureau', 'university', 'institute', 'laboratory']):
                # Check if this looks like author info (multiple names, institutions)
                if len(content.split(',')) >= 3:  # Likely author info
                    authors_text = content
                    authors, institutions = parse_authors_and_institutions(content)
                    break
    
    return authors, institutions, authors_text


def parse_authors_and_institutions(text: str) -> Tuple[List[str], List[str]]:
    """
    Parse authors and institutions from a text string.
    
    Args:
        text: Text containing author and institution information
        
    Returns:
        Tuple of (authors_list, institutions_list)
    """
    authors = []
    institutions = []
    
    # Clean the text
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = text.strip()
    
    # Common patterns for separating authors and institutions
    # Pattern 1: Authors, Institution
    # Pattern 2: Authors<br>Institution
    # Pattern 3: Authors\nInstitution
    
    # Split by common separators
    parts = re.split(r'<br\s*/?>|\n|;', text)
    
    if len(parts) >= 2:
        # First part likely contains authors
        author_part = parts[0].strip()
        # Second part likely contains institution
        institution_part = parts[1].strip()
        
        # Extract authors (split by commas, 'and', '&')
        author_names = re.split(r',\s*|\s+and\s+|\s*&\s*', author_part)
        authors = [name.strip() for name in author_names if name.strip()]
        
        # Extract institutions
        institutions = [institution_part]
        
        # Check for additional institutions in remaining parts
        for part in parts[2:]:
            part = part.strip()
            if part and any(word in part.lower() for word in ['university', 'institute', 'laboratory', 'consulting', 'bureau', 'company', 'inc', 'ltd']):
                institutions.append(part)
    
    else:
        # Single line - try to separate authors and institutions
        # Look for patterns like "Name1, Name2, Institution"
        if ',' in text:
            # Find the last comma that's followed by institution-like text
            parts = text.split(',')
            if len(parts) >= 2:
                # Assume last part is institution, rest are authors
                institution_part = parts[-1].strip()
                author_part = ','.join(parts[:-1]).strip()
                
                # Extract authors
                author_names = re.split(r',\s*|\s+and\s+|\s*&\s*', author_part)
                authors = [name.strip() for name in author_names if name.strip()]
                
                # Extract institutions
                institutions = [institution_part]
    
    # Clean up results
    authors = [author for author in authors if len(author) > 1 and not author.isdigit()]
    institutions = [inst for inst in institutions if len(inst) > 3]
    
    return authors, institutions


def extract_paper_metadata(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract comprehensive paper metadata including title, authors, and institutions.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with paper metadata
    """
    # Extract title
    title = extract_paper_title(chunks)
    
    # Extract authors and institutions
    authors, institutions, authors_text = extract_authors_and_institutions(chunks)
    
    # Look for additional metadata
    abstract = ""
    keywords = []
    doi = ""
    
    for chunk in chunks:
        content = chunk.get('content', '')
        content_type = chunk.get('content_type', '')
        
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        
        # Extract abstract
        if content_type == 'paragraph' and 'abstract' in content.lower()[:50]:
            abstract = content
        
        # Extract keywords
        if 'keywords' in content.lower() and ':' in content:
            keyword_part = content.split(':', 1)[1] if ':' in content else content
            keywords = [kw.strip() for kw in keyword_part.split(',') if kw.strip()]
        
        # Extract DOI
        if 'doi:' in content.lower():
            doi_match = re.search(r'doi:\s*([^\s]+)', content, re.IGNORECASE)
            if doi_match:
                doi = doi_match.group(1)
    
    return {
        'title': title,
        'authors': authors,
        'institutions': institutions,
        'authors_text': authors_text,
        'abstract': abstract,
        'keywords': keywords,
        'doi': doi
    }


def extract_paper_title(chunks: List[Dict[str, Any]]) -> str:
    """
    Extract the paper title from the chunks.
    Looks for the first h1 heading or falls back to other methods.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Paper title as string
    """
    # First, try to find an h1 heading
    for chunk in chunks:
        if (chunk.get('content_type') == 'heading' and 
            chunk.get('level') == 1 and 
            chunk.get('content')):
            return chunk['content'].strip()
    
    # Fallback: look for any heading
    for chunk in chunks:
        if (chunk.get('content_type') == 'heading' and 
            chunk.get('content')):
            return chunk['content'].strip()
    
    # Final fallback: use first non-empty content
    for chunk in chunks:
        if chunk.get('content') and chunk['content'].strip():
            return chunk['content'].strip()[:100] + "..."  # Truncate if too long
    
    return "Untitled Paper"


def convert_chunk_to_database_format(
    chunk: Dict[str, Any], 
    paper_metadata: Dict[str, Any], 
    chunk_id: int
) -> Dict[str, Any]:
    """
    Convert a single chunk to database format.
    
    Args:
        chunk: The chunk dictionary from JSON
        paper_metadata: Dictionary with paper metadata (title, authors, institutions, etc.)
        chunk_id: The ID for this chunk
        
    Returns:
        Database-ready dictionary
    """
    content = chunk.get('content', '').strip()
    
    # Determine the type based on content_type
    content_type = chunk.get('content_type', 'unknown')
    type_mapping = {
        'heading': 'header',
        'paragraph': 'paragraph',
        'list': 'list',
        'table': 'table',
        'image': 'image',
        'code_block': 'code',
        'quote': 'quote',
        'form': 'form',
        'navigation': 'navigation',
        'footer': 'footer',
        'divider': 'divider',
        'unknown': 'unknown'
    }
    
    chunk_type = type_mapping.get(content_type, 'unknown')
    
    # Build metadata - ChromaDB compatible (only primitive types)
    # Extract html_class from attributes.class if available
    html_class = ''
    if chunk.get('attributes', {}).get('class'):
        # If class is a list, join it; if it's a string, use as is
        class_value = chunk['attributes']['class']
        if isinstance(class_value, list):
            html_class = ' '.join(class_value)
        else:
            html_class = str(class_value)
    
    metadata = {
        'name': paper_metadata['title'],
        'type': chunk_type,
        'html_class': html_class,
        'token_count': count_tokens(content),
        'tag_name': chunk.get('tag_name', ''),
        'position': chunk.get('position', 0)
    }
    
    # Add level for headings
    if content_type == 'heading' and chunk.get('level'):
        metadata['level'] = chunk['level']
    
    # Add list_type for lists
    if content_type == 'list' and chunk.get('list_type'):
        metadata['list_type'] = chunk['list_type']
    
    # Add table_info for tables (ChromaDB compatible)
    if content_type == 'table' and chunk.get('table_info'):
        # Only add primitive values from table_info
        table_info = chunk['table_info']
        if 'merged_chunks' in table_info:
            metadata['merged_chunks'] = table_info['merged_chunks']  # This is an int
        # Skip original_positions as it's a list and ChromaDB doesn't accept lists
    
    return {
        'id': chunk_id,
        'content': content,
        'metadata': metadata
    }


def merge_consecutive_table_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive chunks with content_type 'table' into a single chunk.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of chunks with consecutive table chunks merged
    """
    if not chunks:
        return chunks
    
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        current_type = current_chunk.get('content_type', '')
        
        # If this is a table chunk, look for consecutive table chunks
        if current_type == 'table':
            table_chunks = [current_chunk]
            table_contents = [current_chunk.get('content', '').strip()]
            table_attributes = current_chunk.get('attributes', {})
            table_tag_name = current_chunk.get('tag_name', '')
            table_position = current_chunk.get('position', 0)
            
            # Look ahead for consecutive table chunks
            j = i + 1
            while j < len(chunks) and chunks[j].get('content_type', '') == 'table':
                table_chunks.append(chunks[j])
                table_contents.append(chunks[j].get('content', '').strip())
                j += 1
            
            # Merge the table chunks
            if len(table_chunks) > 1:
                # Combine content with separators
                merged_content = ' | '.join([content for content in table_contents if content])
                
                # Create merged chunk
                merged_chunk = {
                    'content_type': 'table',
                    'content': merged_content,
                    'tag_name': table_tag_name,
                    'attributes': table_attributes,
                    'position': table_position,
                    'table_info': {
                        'merged_chunks': len(table_chunks),
                        'original_positions': [chunk.get('position', 0) for chunk in table_chunks]
                    }
                }
                
                merged_chunks.append(merged_chunk)
                print(f"Merged {len(table_chunks)} consecutive table chunks into one")
            else:
                # Single table chunk, keep as is
                merged_chunks.append(current_chunk)
            
            # Skip the chunks we've processed
            i = j
        else:
            # Non-table chunk, keep as is
            merged_chunks.append(current_chunk)
            i += 1
    
    return merged_chunks


def filter_empty_content(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out chunks with empty or whitespace-only content.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Filtered list of chunks
    """
    filtered_chunks = []
    
    for chunk in chunks:
        content = chunk.get('content', '').strip()
        
        # Keep chunks with actual content
        if content:
            # Also keep structural elements like dividers even if they have minimal content
            if (chunk.get('content_type') == 'divider' or 
                chunk.get('content_type') == 'image' or
                len(content) > 0):
                filtered_chunks.append(chunk)
    
    return filtered_chunks


def convert_json_to_database_format(
    input_file: str, 
    output_file: Optional[str] = None,
    include_empty: bool = False,
    merge_tables: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert JSON file of parsed HTML chunks to database format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional)
        include_empty: Whether to include chunks with empty content
        
    Returns:
        List of database-ready dictionaries
    """
    print(f"Loading chunks from: {input_file}")
    
    # Load the JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        return []
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Extract comprehensive paper metadata
    try:
        paper_metadata = extract_paper_metadata(chunks)
        paper_title = paper_metadata['title']
        
        print(f"Paper title: {paper_title}")
        print(f"Authors: {', '.join(paper_metadata['authors']) if paper_metadata['authors'] else 'Not found'}")
        print(f"Institutions: {', '.join(paper_metadata['institutions']) if paper_metadata['institutions'] else 'Not found'}")
    except Exception as e:
        print(f"Warning: Error extracting paper metadata: {e}")
        print("Using fallback metadata extraction...")
        
        # Fallback to simple title extraction
        paper_title = extract_paper_title(chunks)
        paper_metadata = {
            'title': paper_title,
            'authors': [],
            'institutions': [],
            'authors_text': '',
            'abstract': '',
            'keywords': [],
            'doi': ''
        }
        
        print(f"Paper title: {paper_title}")
        print("Authors: Not found (using fallback)")
        print("Institutions: Not found (using fallback)")
    
    # Filter out empty content if requested
    if not include_empty:
        original_count = len(chunks)
        chunks = filter_empty_content(chunks)
        print(f"Filtered out {original_count - len(chunks)} empty chunks")
    
    # Merge consecutive table chunks
    if merge_tables:
        original_table_count = sum(1 for chunk in chunks if chunk.get('content_type') == 'table')
        chunks = merge_consecutive_table_chunks(chunks)
        final_table_count = sum(1 for chunk in chunks if chunk.get('content_type') == 'table')
        if original_table_count != final_table_count:
            print(f"Table merging: {original_table_count} table chunks → {final_table_count} table chunks")
    else:
        print("Table merging disabled")
    
    # Convert chunks to database format
    database_chunks = []
    chunk_id = 1
    
    for chunk in chunks:
        try:
            db_chunk = convert_chunk_to_database_format(chunk, paper_metadata, chunk_id)
            database_chunks.append(db_chunk)
            chunk_id += 1
        except Exception as e:
            print(f"Warning: Error converting chunk {chunk_id}: {e}")
            # Skip this chunk and continue
            continue
    
    print(f"Converted {len(database_chunks)} chunks to database format")
    
    # Save to output file if specified
    if output_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(database_chunks, f, indent=2, ensure_ascii=False)
            
            print(f"Saved database format to: {output_file}")
        except Exception as e:
            print(f"Error saving to {output_file}: {e}")
    
    return database_chunks


def print_database_summary(database_chunks: List[Dict[str, Any]]):
    """
    Print a summary of the database chunks.
    
    Args:
        database_chunks: List of database-ready dictionaries
    """
    if not database_chunks:
        print("No chunks to summarize.")
        return
    
    print("\n=== Database Format Summary ===")
    print(f"Total chunks: {len(database_chunks)}")
    
    # Count by type
    type_counts = {}
    total_tokens = 0
    
    for chunk in database_chunks:
        chunk_type = chunk['metadata']['type']
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        total_tokens += chunk['metadata']['token_count']
    
    print(f"Total tokens: {total_tokens}")
    print("\nChunks by type:")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type}: {count}")
    
    # Show sample chunk
    if database_chunks:
        print(f"\nSample chunk (ID: {database_chunks[0]['id']}):")
        sample = database_chunks[0]
        print(f"  Type: {sample['metadata']['type']}")
        print(f"  Content: {sample['content'][:100]}{'...' if len(sample['content']) > 100 else ''}")
        print(f"  Tokens: {sample['metadata']['token_count']}")
        print(f"  Position: {sample['metadata']['position']}")


def main():
    """Main function to run the converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert parsed HTML chunks to database format')
    parser.add_argument('input_file', help='Input JSON file with parsed chunks')
    parser.add_argument('-o', '--output', help='Output JSON file for database format')
    parser.add_argument('--include-empty', action='store_true', help='Include chunks with empty content')
    parser.add_argument('--no-merge-tables', action='store_true', help='Disable merging of consecutive table chunks')
    parser.add_argument('--summary', action='store_true', help='Print summary of converted data')
    
    args = parser.parse_args()
    
    # Convert the file
    database_chunks = convert_json_to_database_format(
        args.input_file, 
        args.output, 
        args.include_empty,
        merge_tables=not args.no_merge_tables
    )
    
    # Print summary if requested
    if args.summary or not args.output:
        print_database_summary(database_chunks)
    
    print(f"\nConversion complete! Generated {len(database_chunks)} database records.")


if __name__ == "__main__":
    # If run directly without arguments, use default file
    if len(sys.argv) == 1:
        input_file = "documents/processed/JSON/NAICS_JSM_Proceedings_Submission_Aug12_chunks.json"
        output_file = "documents/processed/database/NAICS_JSM_Proceedings_Submission_Aug12_database.json"
        
        print("No arguments provided, using default files:")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print()
        
        database_chunks = convert_json_to_database_format(input_file, output_file)
        print_database_summary(database_chunks)
    else:
        main() 