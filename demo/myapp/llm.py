# llm.py
from llama_index.llms.ollama import Ollama
def get_llm_response(question):
    llm = Ollama(model="vistral", request_timeout=120.0, max_new_tokens=4000) # get_llm()
    response = llm.complete(question)    
    return response
