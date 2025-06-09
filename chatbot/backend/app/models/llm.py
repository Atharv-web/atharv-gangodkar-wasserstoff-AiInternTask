import google.generativeai as genai
from langchain_ollama import ChatOllama
import os
import json
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model='qwen2.5:14b')

def get_answer_and_themes(query, retrieved_docs):

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

TASKS:
1. Answer the user's query based on the formatted context. This context consists of the metadata and the retreived documents from the vector database.
Analyse the retreived documents and answer the users question.

2. Identify the theme/topic based on the formatted context. This contains the documents that are retreived from the vector database and the metadata related to it.
Analyse this formatted context to identify the theme/topic.  

Remember this: 
Use the metadata from the formatted context to get the name of the document for DOCUMENT ID by checking the title.
Use the metadata from the formatted context to get the CITATION by checking the page number.
Use the context from formatted context to get the EXTRACTED ANSWER.

Respond in **valid JSON** using the format:
{{
  "answers": [
    {{
      "document_id": ...,
      "extracted_answer": ...,
      "citation": ...,
    }},
    ...
  ],
  "themes": [
    {{
      "theme": "Theme Name",
      "summary": "Brief summary of the theme...",
    }},
    ...
  ]
}}
"""

    # Run Model call
    response = model.invoke(prompt)

    try:
        return json.loads(response.content)
    except Exception as e:
        return {
            "answers": [],
            "themes": [{
                "theme": "ParsingError",
                "summary": f"Could not parse model response. Error: {str(e)}",
            }]
        }
