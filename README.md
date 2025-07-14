# DocParser

A comprehensive document processing pipeline that converts PDFs into searchable, queryable content. Gemini is used for parsing documents and OpenAI is used for the agentic workflows. 

## Overview

DocParser is a multi-stage document processing system that:

1. Converts PDFs to HTML using Google's Gemini AI (top performer for doc parsing: [OpenCompass](https://rank.opencompass.org.cn/leaderboard-llm))
2. Parses HTML into structured JSON with content type classification
3. Chunks content for optimal vector database storage with relevant metadata
4. Builds a ChromaDB vector database for semantic search available as an MCP server
5. Provides an OpenAI agent interface for querying documents

## Architecture

The system follows a modular design with clear separation of concerns across document processing and querying:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Input     │    │   Processing    │    │   Chroma        │
│   Documents     │───►│   Pipeline      │───►│   Vector DB     │
│                 │    │   (4 stages)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       ▲
                                ▼                       │
                       ┌─────────────────┐              │
                       │   MCP Server    │◄─────────────┘
                       │   (mcp_server)  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Agent         │    │   OpenAI API    │
                       │   (run_agent)   │◄──►│   Embeddings    │
                       └─────────────────┘    └─────────────────┘
```


### Core Components

- **`src/doc_parser.py`**: Uses Gemini AI to convert PDFs to HTML
- **`src/html_parser.py`**: Parses HTML into structured JSON with content type classification
- **`src/chunker.py`**: Converts JSON into database-ready chunks with metadata
- **`src/build_chroma.py`**: Builds and manages ChromaDB collections
- **`download_pdfs.py`**: Downloads PDFs to the documents folder for processing
- **`run_processing.py`**: Orchestrates the entire document processing pipeline
- **`run_agent.py`**: OpenAI agent interface for querying processed documents
- **`server/mcp_server.py`**: Model Context Protocol server for document interactions

### Description of Workflow

The strategy behind using HTML as a first layer is to rely on a powerful LLM (Gemini) to identify the language within the paper that might indicate meaningful text (titles, abstracts, sections) as well as multimodal elements (tables, images). From this format of reliable HTML, we parse into a JSON object containing key attributes. From there, another JSON object is formed that consists of the chunks of data (id, document, metadata) that will be inserted into a vector database. Once prepared, the vector database is made available via model context protocol for an OpenAI agent that uses the database and additional instructions to make queries and provide informed answers. 


### Repository Architecture

```
DocParser/
├── 📁 src/                         ## Document Processing Pipeline
│   ├── doc_parser.py                # 1.PDF to HTML conversion using Gemini AI
│   ├── html_parser.py               # 2. HTML parsing with content classification
│   ├── chunker.py                   # 3. JSON to database format conversion
│   ├── build_chroma.py              # 4. ChromaDB collection management
│   └── run_processing.log           
│
├── 📁 server/                      ## MCP Server components
│   ├── mcp_server.py               # MCP server for ChromaDB interactions
│   └── agent_qa_log.json           # Record of agent questions and answers
│
├── 📁 documents/                   # Document storage and processing
│   ├── 📁 pdfs/                    # Input PDF files
│   └── 📁 processed/               # Processed document outputs
│       ├── 📁 HTML/                # Generated HTML files
│       ├── 📁 JSON/                # Parsed JSON chunks
│       ├── 📁 database/            # Database-ready chunks
│
├── 🚀 run_processing.py            # Main pipeline orchestrator
├── 📥 download_pdfs.py             # PDF download utility
├── 🤖 run_agent.py                 # OpenAI agent interface
├── 🔧 clients.py                   # Client utilities
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # Project documentation
```


## Installation

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DocParser
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

### Quick Start

1. **Place PDFs in the documents folder**
   ```bash
   cp your_documents.pdf documents/pdfs/
   ```

   or copy and paste the URLs of the PDFs you want to process into download_pdfs.py and run

   ```bash
   python download_pdfs.py
   ```

2. **Run the processing pipeline**
   ```bash
   python run_processing.py
   ```

3. **Start the agent interface**
   ```bash
   python run_agent.py
   ```

View the questions and answers in the terminal or the server/agent_qa_log.json!
