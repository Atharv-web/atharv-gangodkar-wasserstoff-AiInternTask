
from langchain_ollama import ChatOllama
import os
import json
from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(model="qwen2.5:14b")

def get_answer_and_themes(query, retrieved_docs):

    formatted_context = "\n".join(
        f" Context: {doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_docs
    )
    
    prompt = f"""

You are a Document Research Chatbot.

A user asked:
"{query}"

Here are relevant excerpts from documents extracted:

{formatted_context}

Your task:
1. Provide a concise answer that is relevant to the query.
2. Each answer, must include:
   - document_id (use title from metadata in formatted_context)
   - extracted_answer (the extracted content from the document)
   - citation (e.g., page number, paragraph reference, from metadata)
3. Identify the theme from the context:
   - Provide the theme title
   - Give a brief summary

Return your answer in this JSON format:

{{
  "answers": [
    {{
      "document_id" : "use information from metadata in formatted_context",
      "extracted_answers" : "the extracted content from the document",
      "citation" : "page number, paragraph reference, from metadata"
    }}
  ],
  "themes": [
    {{
      "theme" : "Theme title",
      "summary" : "Brief Summary abt the extracted answer and theme"
    }}
  ]
}}

"""
    # # Run Model call
    response = model.invoke(prompt)
    try:
        # Clean the response content and parse JSON
        content = response.content.strip()
        # Remove any markdown code block indicators if present
        content = content.replace('```json', '').replace('```', '').strip()
        result = json.loads(content)
        return result
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return {
            "answers": [],
            "themes": [{
                "theme": "Error",
                "summary": f"Failed to parse model response as JSON: {str(e)}",
            }]
        }
