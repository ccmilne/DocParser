import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from agents.mcp import MCPServerStdio, MCPServer
from agents import Agent, Runner, set_default_openai_key

load_dotenv()

set_default_openai_key(os.getenv("OPENAI_API_KEY"))


def save_qa_data(qa_data: list, filename: str = "agent_qa_log.json"):
    """
    Save Q&A data to a JSON file, overwriting the previous content.
    
    Args:
        qa_data: List of dictionaries containing question-answer pairs
        filename: Name of the file to save to
    """
    # Create the output directory if it doesn't exist
    output_dir = Path("server")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    # Prepare the data structure
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(qa_data),
        "qa_pairs": qa_data
    }
    
    # Write to file, overwriting previous content
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Q&A data saved to: {filepath}")


async def run(server: MCPServer):
    """
    Runs the OpenAI Agent with the provided MCP server.
    """
    agent = Agent(
        name="Assistant",
        instructions="""You are a helpful assistant that can answer questions about the documents in the database.
        
    Use the following tools to answer questions:    
    - chroma_list_collections: lists the names of all the papers in the database for further research
    - chroma_get_collection_info: gets the information about a specific collection
    - chroma_query_documents: queries the documents in a specific collection
    - chroma_get_documents: gets the documents in a specific collection

    
    """,
        mcp_servers=[server],
    )
    
    # List to store all Q&A pairs
    qa_data = []
    
    # Helper function to run a question and capture the response
    async def ask_question(question: str, question_type: str = "general"):
        print(f"\n{'='*60}")
        print(f"Question ({question_type}): {question}")
        print(f"{'='*60}")
        
        response = await Runner.run(starting_agent=agent, input=question)
        print(f"Response: {response}")
        
        # Store the Q&A pair
        qa_pair = {
            "question": question,
            "question_type": question_type,
            "response": response.final_output,
            "timestamp": datetime.now().isoformat()
        }
        qa_data.append(qa_pair)
        
        return response
    
    # Starter question to ensure server is responsive
    await ask_question(
        "Test the server functionality first, then let me know if you are ready to answer questions.",
        "server_test"
    )
    
    # Basic question:
    await ask_question(
        "What are the titles of all the papers in the database?",
        "basic"
    )
    
    # Test ability to collection metadata from the collections
    await ask_question(
        "What are the names of each of the collections, and how many documents are in each collection?",
        "metadata"
    )
    
    # More complex question:
    await ask_question(
        "What does the paper on Ensemble Retrieval Strategies aim to research?",
        "research_analysis"
    )
    
    # Test ability to extract table information from a paper
    await ask_question(
        "What do testing results look like for the paper on Ensemble Retrieval Strategies?",
        "table_extraction"
    )
    
    # Hard questions
    await ask_question(
        "What is the title of the paper that has the highest number of citations?",
        "advanced_analysis"
    )
    
    await ask_question(
        "What is the best performing model in the Attention is all you need paper for EN-FR?",
        "advanced_analysis"
    )
    
    # Save all Q&A data to file
    save_qa_data(qa_data)
    
    print(f"\n{'='*60}")
    print(f"Session completed. {len(qa_data)} questions processed.")
    print(f"{'='*60}")
    
    
    
async def test():
    """
    Defines the MCP server and runs the agent.
    """
    params = {
        "command": "python",
        "args": ["server/mcp_server.py"],
        "env": {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        },
    }
    timeout = 30.0
    
    async with MCPServerStdio(params, client_session_timeout_seconds=timeout) as server:
        await run(server)


if __name__ == "__main__":
    asyncio.run(test())