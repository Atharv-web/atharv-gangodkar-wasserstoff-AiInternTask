from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ["GEMINI_API_KEY"]

def model_call():
  model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key = API_KEY)
  return model

def get_response(session_history):
  model = model_call()
  for entry in session_history:
    if "retrieved_context" in entry:
        context = entry["retrieved_context"]


  prompt = f"""
You are an assistant for question-answering tasks related to the users uploaded documents. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Keep the answer concise. 
Context: {context}

"""
  session_history.append(SystemMessage(content=prompt))


  try:
    response = model.invoke(session_history)
    return response.content
  except Exception as e:
    return f"Loading Error Bitch!!, btw the error is :{e}"