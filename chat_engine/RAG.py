from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import QueryBundle
from llama_index.llms.gemini import Gemini
import os
import re
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings

from dotenv import load_dotenv

from prompts import gen_qa_prompt, gen_rag_answer, formatted_context, intent_classification_prompt, text_summarization_prompt

load_dotenv()


class RAG:
    def __init__(
        self,
        model_name="vistral-legal-chat",
        embedding_model="/teamspace/studios/this_studio/bge-m3",
        dataset_name="Vietnamese-law-RAG-semantic-chuning"
    ):
        self.model = Ollama(model=model_name, request_timeout=120.0)
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = self.embed_model
        Settings.llm = self.model

        ## get vector store 
        self.activeloop_id = "hunter"
        self.dataset_name = dataset_name
        self.dataset_path = f"hub://{self.activeloop_id}/{self.dataset_name}"

        self.vector_store = DeepLakeVectorStore(
            dataset_path=self.dataset_path,
            overwrite=False,
        )

        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        ## get retrievers
        self.vector_retriever = self.index.as_retriever(
            similarity_top_k=5,
        )
        # bm25 retriever
        source_nodes = self.index.as_retriever(similarity_top_k=200000).retrieve("test")
        nodes = [x.node for x in source_nodes]
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

        # self.vector_retriever, self.bm25
        self.retrievers = [self.vector_retriever, self.bm25]

        ## metadata replacement and reranker
        self.replacement = MetadataReplacementPostProcessor(target_metadata_key="window")
        self.cohere_rerank = CohereRerank(model="rerank-multilingual-v2.0", api_key=os.getenv('COHERE_API_KEY'), top_n=3)  # remain top 3 relevant
    
    def generate_queries(self, query_str, num_queries=3, llm=Gemini(model_name="models/gemini-1.5-flash")):
        prompt = gen_qa_prompt.format(
            num_queries=num_queries, query=query_str
        )
        response = llm.complete(prompt)
        queries = response.text.split("\n")
        return queries

    def retrieve(self, query_str):
        # generate_queries
        queries = self.generate_queries(query_str)

        ## get relevant nodes
        nodes = []
        for retriever in self.retrievers:
            # Retrieve nodes for both queries and the original query
            for query in queries + [query_str]:
                if len(query) > 0:
                    retrieved_nodes = retriever.retrieve(str(query))
                    nodes.extend([node for node in retrieved_nodes if node.score >= 0.7])
        
        # replace metadata
        nodes = self.replacement.postprocess_nodes(nodes)

        # rerank
        final_nodes = self.cohere_rerank.postprocess_nodes(
            nodes, QueryBundle(query_str)
        )

        return final_nodes

    def answer(self, query_str):
        final_nodes = self.retrieve(query_str)

        context = "\n".join([
            formatted_context.format(
                law=node.metadata["file_name"].split(".")[0], 
                content=node.get_content()
            )
            for node in final_nodes
        ])

        response = self.model.complete(
            gen_rag_answer.format(query_str=query_str, context_str=context)
        )

        return response

    def base_answer(self, query_str):
        query_engine = self.index.as_query_engine(similiarity_top_k=6)
        return query_engine.query(query_str)


    def response(self, query_str):
        final_nodes = self.retrieve(query_str)

        context = "\n".join([
            formatted_context.format(
                law=node.metadata["file_name"].split(".")[0], 
                content=node.get_content()
            )
            for node in final_nodes
        ])

        response = self.model.complete(
            gen_rag_answer.format(query_str=query_str, context_str=context)
        )

        sources = [
            node.metata['file_name']
            for node in final_nodes
        ]

        return response, sources

    # def intent_classification(self, query_str):
    #     """
    #     consider whether the user's query is legal-related or just normal question
    #     """
        
    #     model = Gemini(model_name="models/gemini-1.5-flash", temperature=0)

    #     max_attempt = 5
    #     current_attempt = 0
    #     while current_attempt < max_attempt:
    #         output = model.complete(intent_classification_prompt).text.strip()
    #         if output in {"0", "1"}:
    #             return int(output)    
    #         else:
    #             current_attempt += 1
        
    #     return 0

    def intent_classification(self, query_str):
        """
        Classify whether the user's query is related to legal issues or just a normal question.
        """
        model = Gemini(model_name="models/gemini-1.5-flash", temperature=0)
        prompt = intent_classification_prompt.format(query_str) 

        max_attempts = 5
        current_attempt = 0
        
        while current_attempt < max_attempts:
            try:
                output = model.complete(prompt).text.strip()
                if output in {"0", "1"}:
                    return int(output)
            except Exception as e:
                print(f"Error during classification attempt {current_attempt + 1}: {e}")
            
            current_attempt += 1
        
        # Log or handle the case when classification fails after max attempts
        print(f"Failed to classify after {max_attempts} attempts.")
        return 0

    def web_search(self, query_str):
        """
        Tavily search tool
        -Tavily API
        """
        pass

    def evaluate_hallucination(self, query_str):
        """
        using gpt or gemini to check hallucination of the answer compared to the original query
        gemini-1.5-flash
        """
        pass

    def summarization(self, text):
        """
        summarize the context to make it shorter, more succin -> help reduce memory abundance
        leverage Gemini-1.5-flash
        """
        model = Gemini(model_name="models/gemini-1.5-flash", temperature=0)
        output = model.complete(text_sum).text
        return output




def format_chatbot_output(input_text):
    # Convert input to string if it isn't already
    text = str(input_text)
    
    # Replace dash followed by a space with a new line and dash
    text = text.replace(" - ", "\n- ")
    text = text.replace("<", "")
    text = text.replace("<<", "")
    text = text.replace(">", "")
    text = text.replace(">>", "")
    text = text.replace("Bạn là một chuyên viên tư vấn pháp luật Việt Nam. Bạn có nhiều năm kinh nghiệm và kiến thức chuyên sâu. Bạn sẽ cung cấp câu trả lời về pháp luật cho các câu hỏi của User.", "")
    text = text.replace("Bạn là một chuyên viên tư vấn pháp luật tại Việt Nam với nhiều năm kinh nghiệm và kiến thức chuyên sâu. Nhiệm vụ của bạn là cung cấp câu trả lời và tư vấn pháp lý cho các câu hỏi của người dùng","")
    # Define a regex pattern to match "a)", "b)", "c)", etc.
    pattern = re.compile(r'(\s[a-zA-Z]\)|\s\d+\)|\s[a-zA-Z]\.|\s\d+\.)')
    
    # Replace matches with new line and the matched character
    formatted_text = pattern.sub(r'\n\1', text)
    
    return formatted_text

def return_answer(model, query):
    
    query_type = model.intent_classification(query)
    answer = ''
    if(query_type==1):
        answer = model.answer(query)
    else:
        print('non-legal related question')
        ollama_model = Ollama(model="vistral-legal-chat", request_timeout=120.0)
        answer = ollama_model.complete(query)
        
    answer = format_chatbot_output(answer)

    return answer

if __name__ == "__main__":
    
    # sudo systemctl stop ollama
    # ollama serve
    rag = RAG(
        model_name="vistral-legal-chat"    # vistral, phogpt, vinallama
    )

    
    
    
