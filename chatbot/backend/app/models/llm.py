from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ['GEMINI_API_KEY']
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash',api_key = API_KEY)

def get_answer_and_themes(query, retrieved_docs):

    formatted_context = "\n".join(
        f" Context: {doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_docs
    )

    # Prompt
    prompt = f"""

You are a Document Research Chatbot.

A user asked:
"{query}"

Here are relevant excerpts from documents extracted:

{formatted_context}

Instructions:
- Carefully read all provided context and metadata.
- Synthesize a single, comprehensive answer to the user's question, using the most relevant information from the context.
- Do NOT provide one answer per document just ONE best answer.
- If possible, indicate the source document's name and citation (from metadata) that supports your answer.
- Also, identify the main theme or topic in discussion.

Return your answer in this JSON format:

{{
    "answers": {{
        "document_id": "...",     
        "extracted_answer": "...",
        "citation": "..."
    }},
    "themes": {{
        "theme": "...",
        "summary": "..."
    }}
}}

"""
    # Run Model call
    response = model.invoke(prompt)
    try:
        answer = response.content
        content = answer.replace('```json', '').replace('```', '').strip()
        result = json.loads(content)
        return result
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "answers": [],
            "themes": [{
                "theme": "Error",
                "summary": f"An unexpected error occurred: {str(e)}",
            }]
        }
