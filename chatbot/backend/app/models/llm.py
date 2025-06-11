import google.generativeai as genai
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ['GEMINI_API_KEY']
# model = ChatOllama(model='qwen2.5:14b')
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',api_key = API_KEY)

def get_answer_and_themes(query, retrieved_docs):
    """
    Generate answers and identify themes using Gemini based on query and context.
    Each document chunk must have: file_path, paragraph_number, chunk (text content).
    """

    formatted_context = "\n".join(
        f" Context: {doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_docs
    )

    # Prompt
    prompt = f"""
You are a Document Research & Theme Identification Chatbot.

A user asked:
"{query}"

Here are relevant excerpts from documents extracted:

{formatted_context}

Remember this: 
Use the metadata from the formatted context to get the name of the document for DOCUMENT ID by checking the title.
Use the metadata from the formatted context to get the CITATION by checking the page number.
Use the context from formatted context to get the EXTRACTED ANSWER.
Use the user query and context from formatted context to identify the theme or topic in discussion

the content u generate must be in this json format:

this format to be followed is:
{{
    "answers": [
        {{
            "document_id": ...,
            "extracted_answers": ...,
            "citation": ...
        }}
    ],
    "themes": [
        {{
            "theme": ...,
            "summary": ...
        }}
    ]
}}
"""
    # Run Model call
    response = model.invoke(prompt)
    try:
        # Clean the response content to ensure it's valid JSON
        content = response.content.strip()
        # Remove any markdown code block indicators if present
        content = content.replace('```json', '').replace('```', '').strip()
        # Parse the JSON
        result = json.loads(content)
        
        # Validate the structure
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")
        if "answers" not in result or "themes" not in result:
            raise ValueError("Missing required fields in response")
            
        return result
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        print(f"Raw response: {response.content}")
        return {
            "answers": [],
            "themes": [{
                "theme": "JSONParsingError",
                "summary": "Failed to parse the model's response into valid JSON format",
            }]
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "answers": [],
            "themes": [{
                "theme": "Error",
                "summary": f"An unexpected error occurred: {str(e)}",
            }]
        }
