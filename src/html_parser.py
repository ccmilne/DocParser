import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from bs4 import BeautifulSoup, Tag, NavigableString


class ContentType(Enum):
    """Enumeration of different content types that can be extracted from HTML."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    DIVIDER = "divider"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    FORM = "form"
    NAVIGATION = "navigation"
    FOOTER = "footer"
    HEADER = "header"
    SIDEBAR = "sidebar"
    UNKNOWN = "unknown"


@dataclass
class ContentChunk:
    """Represents a chunk of content with its type and metadata."""
    content_type: ContentType
    content: str
    tag_name: str
    attributes: Dict[str, str]
    level: Optional[int] = None  # For headings (h1=1, h2=2, etc.)
    list_type: Optional[str] = None  # For lists (ul, ol)
    table_info: Optional[Dict[str, Any]] = None  # For tables
    position: Optional[int] = None  # Position in the document
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ContentChunk to a JSON-serializable dictionary."""
        return {
            'content_type': self.content_type.value,
            'content': self.content,
            'tag_name': self.tag_name,
            'attributes': self.attributes,
            'level': self.level,
            'list_type': self.list_type,
            'table_info': self.table_info,
            'position': self.position
        }
    
    def __repr__(self) -> str:
        """String representation of the ContentChunk."""
        return f"ContentChunk(type={self.content_type.value}, tag={self.tag_name}, content='{self.content[:50]}{'...' if len(self.content) > 50 else ''}')"


class HTMLContentParser:
    """
    A parser that separates HTML content into chunks based on content type.
    Each chunk represents a logical unit of content like a paragraph, heading, list, etc.
    """
    
    def __init__(self):
        # Tags that represent different content types
        self.heading_tags = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
        self.paragraph_tags = {'p'}
        self.list_tags = {'ul', 'ol', 'li'}
        self.table_tags = {'table', 'thead', 'tbody', 'tr', 'th', 'td'}
        self.divider_tags = {'hr'}
        self.image_tags = {'img'}
        self.code_tags = {'pre', 'code'}
        self.quote_tags = {'blockquote', 'q'}
        self.form_tags = {'form', 'input', 'textarea', 'select', 'button'}
        self.nav_tags = {'nav'}
        self.footer_tags = {'footer'}
        self.header_tags = {'header'}
        
        # Tags to ignore (usually structural or styling)
        self.ignore_tags = {
            'script', 'style', 'meta', 'link', 'title', 'head', 'html', 'body',
            'div', 'span', 'section', 'article', 'aside', 'main'
        }
        
        # Tags that might contain content but are usually structural
        self.structural_tags = {'div', 'section', 'article', 'aside', 'main'}
    
    def parse_html(self, html_string: str) -> List[ContentChunk]:
        """
        Parse HTML string into content chunks based on content type.
        
        Args:
            html_string: The HTML string to parse
            
        Returns:
            List of ContentChunk objects representing different content types
        """
        # Clean the HTML string
        html_string = self._clean_html_string(html_string)
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_string, 'html.parser')
        
        # Extract chunks
        chunks = []
        position = 0
        
        for element in soup.find_all(True):
            chunk = self._process_element(element, position)
            if chunk:
                chunks.append(chunk)
                position += 1
        
        return chunks
    
    def _clean_html_string(self, html_string: str) -> str:
        """Clean and normalize the HTML string."""
        # Remove any leading/trailing whitespace
        html_string = html_string.strip()
        
        # If the string starts with ```html, remove it
        if html_string.startswith('```html'):
            html_string = html_string[7:]
        
        # If the string ends with ```, remove it
        if html_string.endswith('```'):
            html_string = html_string[:-3]
        
        return html_string.strip()
    
    def _process_element(self, element: Tag, position: int) -> Optional[ContentChunk]:
        """Process a single HTML element and return a ContentChunk if applicable."""
        tag_name = element.name.lower()
        
        # Skip ignored tags
        if tag_name in self.ignore_tags:
            return None
        
        # Get element attributes
        attributes = dict(element.attrs) if element.attrs else {}
        
        # Process based on tag type
        if tag_name in self.heading_tags:
            return self._process_heading(element, attributes, position)
        elif tag_name in self.paragraph_tags:
            return self._process_paragraph(element, attributes, position)
        elif tag_name in self.list_tags:
            return self._process_list(element, attributes, position)
        elif tag_name in self.table_tags:
            return self._process_table(element, attributes, position)
        elif tag_name in self.divider_tags:
            return self._process_divider(element, attributes, position)
        elif tag_name in self.image_tags:
            return self._process_image(element, attributes, position)
        elif tag_name in self.code_tags:
            return self._process_code(element, attributes, position)
        elif tag_name in self.quote_tags:
            return self._process_quote(element, attributes, position)
        elif tag_name in self.form_tags:
            return self._process_form(element, attributes, position)
        elif tag_name in self.nav_tags:
            return self._process_navigation(element, attributes, position)
        elif tag_name in self.footer_tags:
            return self._process_footer(element, attributes, position)
        elif tag_name in self.header_tags:
            return self._process_header(element, attributes, position)
        elif tag_name in self.structural_tags:
            return self._process_structural(element, attributes, position)
        else:
            return self._process_unknown(element, attributes, position)
    
    def _process_heading(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process heading elements (h1-h6)."""
        level = int(element.name[1])  # Extract level from tag name (h1 -> 1)
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.HEADING,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            level=level,
            position=position
        )
    
    def _process_paragraph(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process paragraph elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.PARAGRAPH,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_list(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process list elements (ul, ol, li)."""
        list_type = element.name if element.name in ['ul', 'ol'] else None
        content = self._extract_list_content(element)
        
        return ContentChunk(
            content_type=ContentType.LIST,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            list_type=list_type,
            position=position
        )
    
    def _process_table(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process table elements."""
        table_info = self._extract_table_info(element)
        content = self._extract_table_content(element)
        
        return ContentChunk(
            content_type=ContentType.TABLE,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            table_info=table_info,
            position=position
        )
    
    def _process_divider(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process divider elements (hr)."""
        return ContentChunk(
            content_type=ContentType.DIVIDER,
            content="---",
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_image(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process image elements."""
        src = attributes.get('src', '')
        alt = attributes.get('alt', '')
        content = f"[Image: {alt}] ({src})" if alt else f"[Image] ({src})"
        
        return ContentChunk(
            content_type=ContentType.IMAGE,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_code(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process code elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.CODE_BLOCK,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_quote(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process quote elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.QUOTE,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_form(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process form elements."""
        content = self._extract_form_content(element)
        
        return ContentChunk(
            content_type=ContentType.FORM,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_navigation(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process navigation elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.NAVIGATION,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_footer(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process footer elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.FOOTER,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_header(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process header elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.HEADER,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _process_structural(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process structural elements that might contain content."""
        content = self._extract_text_content(element)
        
        if content.strip():
            return ContentChunk(
                content_type=ContentType.UNKNOWN,
                content=content,
                tag_name=element.name,
                attributes=attributes,
                position=position
            )
        return None
    
    def _process_unknown(self, element: Tag, attributes: Dict[str, str], position: int) -> ContentChunk:
        """Process unknown elements."""
        content = self._extract_text_content(element)
        
        return ContentChunk(
            content_type=ContentType.UNKNOWN,
            content=content,
            tag_name=element.name,
            attributes=attributes,
            position=position
        )
    
    def _extract_text_content(self, element: Tag) -> str:
        """Extract text content from an element, handling nested elements."""
        if isinstance(element, NavigableString):
            return str(element).strip()
        
        # Get direct text content
        text_parts = []
        for content in element.contents:
            if isinstance(content, NavigableString):
                text = str(content).strip()
                if text:
                    text_parts.append(text)
            elif content.name in self.ignore_tags:
                # Skip ignored tags but get their text content
                text = self._extract_text_content(content)
                if text:
                    text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _extract_list_content(self, element: Tag) -> str:
        """Extract content from list elements."""
        if element.name == 'li':
            return self._extract_text_content(element)
        else:
            # For ul/ol, get all list items
            items = []
            for li in element.find_all('li', recursive=False):
                item_text = self._extract_text_content(li)
                if item_text:
                    items.append(f"• {item_text}")
            return '\n'.join(items)
    
    def _extract_table_content(self, element: Tag) -> str:
        """Extract content from table elements."""
        if element.name in ['th', 'td']:
            return self._extract_text_content(element)
        else:
            # For table, thead, tbody, tr, get all cells
            cells = []
            for cell in element.find_all(['th', 'td']):
                cell_text = self._extract_text_content(cell)
                if cell_text:
                    cells.append(cell_text)
            return ' | '.join(cells)
    
    def _extract_table_info(self, element: Tag) -> Dict[str, Any]:
        """Extract table metadata."""
        if element.name == 'table':
            rows = element.find_all('tr')
            cols = 0
            if rows:
                cols = max(len(row.find_all(['th', 'td'])) for row in rows)
            
            return {
                'rows': len(rows),
                'columns': cols,
                'has_header': bool(element.find('thead') or element.find('th'))
            }
        return {}
    
    def _extract_form_content(self, element: Tag) -> str:
        """Extract content from form elements."""
        if element.name == 'form':
            inputs = []
            for input_elem in element.find_all(['input', 'textarea', 'select']):
                input_type = input_elem.get('type', 'text')
                placeholder = input_elem.get('placeholder', '')
                name = input_elem.get('name', '')
                inputs.append(f"[{input_type}: {name}] {placeholder}")
            return ' | '.join(inputs)
        else:
            return self._extract_text_content(element)
    
    def get_chunks_by_type(self, chunks: List[ContentChunk], content_type: ContentType) -> List[ContentChunk]:
        """Get all chunks of a specific content type."""
        return [chunk for chunk in chunks if chunk.content_type == content_type]
    
    def get_chunks_summary(self, chunks: List[ContentChunk]) -> Dict[str, int]:
        """Get a summary of chunk types and their counts."""
        summary = {}
        for chunk in chunks:
            chunk_type = chunk.content_type.value
            summary[chunk_type] = summary.get(chunk_type, 0) + 1
        return summary
    
    def print_chunks(self, chunks: List[ContentChunk], show_attributes: bool = False):
        """Print chunks in a readable format."""
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Type: {chunk.content_type.value}")
            print(f"Tag: {chunk.tag_name}")
            if chunk.level:
                print(f"Level: {chunk.level}")
            if chunk.list_type:
                print(f"List Type: {chunk.list_type}")
            if chunk.table_info:
                print(f"Table Info: {chunk.table_info}")
            if show_attributes and chunk.attributes:
                print(f"Attributes: {chunk.attributes}")
            print(f"Content: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")


def parse_html_content(html_string: str) -> List[ContentChunk]:
    """
    Convenience function to parse HTML content into chunks.
    
    Args:
        html_string: The HTML string to parse
        
    Returns:
        List of ContentChunk objects
    """
    parser = HTMLContentParser()
    return parser.parse_html(html_string)


def chunks_to_json(chunks: List[ContentChunk]) -> List[Dict[str, Any]]:
    """
    Convert a list of ContentChunk objects to JSON-serializable format.
    
    Args:
        chunks: List of ContentChunk objects
        
    Returns:
        List of dictionaries that can be serialized to JSON
    """
    return [chunk.to_dict() for chunk in chunks]


def save_chunks_to_json(chunks: List[ContentChunk], filename: str):
    """
    Save a list of ContentChunk objects to a JSON file.
    
    Args:
        chunks: List of ContentChunk objects
        filename: Output JSON filename
    """
    import json
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to JSON-serializable format
    json_data = chunks_to_json(chunks)
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False) 