from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
import os

from dotenv import load_dotenv

load_dotenv()

## embedding model
def get_embedding_model(model_name="/teamspace/studios/this_studio/bge-small-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    return embed_model

# generator model
def get_llm(model_name="llama3-8b-8192"):
    llm = Groq(model=model_name, api_key=os.getenv("GROQ_API"), temperature=0.8)
    return llm

## get deeplake vector database
def get_vector_database(id, dataset_name):
    my_activeloop_org_id = id # "hunter"
    my_activeloop_dataset_name = dataset_name # "Vietnamese-law-RAG"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=False,
    )
    return vector_store

def get_index(vector_store):
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

## -------------------------------
## query generation / rewriting
from llama_index.core import PromptTemplate

query_str = "Vượt đèn đỏ sẽ bị gì?"


prompt_for_generating_query = PromptTemplate(
    "Bạn là một trợ lý tuyệt vời trong việc tạo ra các câu truy vấn (query) dựa trên "
    "một câu truy vấn input. Tạo ra {num_queries} truy vấn tìm kiếm (search query), mỗi câu 1 dòng " 
    "liên quan đến câu truy vấn (query) đầu vào được cung cấp dưới đây. Hãy nhớ, trả lời bằng tiếng Việt nhé!!!\n"
    "Only return generated queries\n"
    "Query: {query}\n" 
    "Queries:\n"
)

def generate_queries(llm, query_str, num_queries=4):
    fmt_prompt = prompt_for_generating_query.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries =  response.text.split("\n")
    return queries

def run_queries(queries, retrievers):
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.retrieve(query))
    
    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, tasks)):
        results_dict[(query, i)] = query_result
    
    return results_dict

## get bm25 retriever
from llama_index.retrievers.bm25 import BM25Retriever

def get_bm25_retriever(index, similarity_top_k=13625):
    source_nodes = index.as_retriever(similarity_top_k=similarity_top_k).retrieve("test")
    nodes = [x.node for x in source_nodes]
    bm25 = BM25Retriever.from_defaults(nodes=nodes)
    return bm25


######################################### test
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core import Settings

# embed_model = get_embedding_model()
# Settings.embed_model = embed_model

# llm = get_llm()
# Settings.llm = llm
# vector_store = get_vector_database("hunter", "Vietnamese-law-RAG")
# index = get_index(vector_store=vector_store)
# vector_retriever = index.as_retriever(similarity_top_k=3)
# bm25_retriever = get_bm25_retriever(index)

# #####################################
# query = "Luật Giáo dục đại học sua doi có bao nhiêu điều?"
# queries = generate_queries(llm, query, num_queries=4)
# queries = [q for q in queries if len(q) > 0]

# print(len(queries))

# retrieved_nodes = run_queries(queries, [vector_retriever, bm25_retriever])

# ## filter nodes
# nodes = []
# l = 0
# for _, ns in retrieved_nodes.items():
#     for n in ns:
#         l += 1
#         if n.score > 0.7:
#             nodes.append(n)

# for node in nodes:
#     print(node)

# ## rerank
# from llama_index.postprocessor.cohere_rerank import CohereRerank

# cohere_rerank = CohereRerank(model="rerank-multilingual-v2.0", api_key=os.getenv('COHERE_API_KEY'), top_n=3)  # remain top 3 relevant

# from llama_index.core.schema import QueryBundle
# final_nodes = cohere_rerank.postprocess_nodes(
#     nodes, QueryBundle(query)
# )

prompt_ = """
Bạn là một trợ lý ảo về tư vấn pháp luật. Nhiệm vụ của bạn là sinh ra câu trả lời dựa vào hướng dẫn được cung cấp. 
Với điểm thể hiện độ liên quan với câu trả lời được sắp xếp từ cao đến thấp. Phía dưới đây chúng tôi sẽ cung cấp nhiệm vụ của bạn và thẻ <INS> sẽ được thêm vào trước mỗi prompt templates. 

Giống như ví dụ sau:
```
<INS>
prompt templates
<QUES>
<REF>
```

Prompt templates này sẽ chứa phần: câu hỏi được kí hiệu bằng thẻ <QUES> và phần tài liệu tham khảo được kí hiệu bằng thẻ <REF>.

# Quy tắc trả lời:
1. Bạn phải dựa vào thông tin của phần tài liệu tham khảo <REF> để đưa ra câu trả lời và không được phép dùng kiến thức sẵn có của bạn.
2. Dựa vào thông tin của phần tài liệu tham khảo <REF> hãy trả lời như thể đây là kiến thức của bạn, không dùng các cụm từ như: dựa vào thông tin bạn cung cấp, dựa vào thông tin dưới đây, dựa vào tài liệu tham khảo,...
3. Nếu bạn không tìm thấy thông tin để trả lời trong phần tài liệu tham khảo <REF> thì hãy trả lời rằng thông tin được cung cấp không có đủ thông tin để trả lời.
4. Nếu người dùng hỏi những câu hỏi chứa nội dung không tiêu cực, không lạnh mạnh hãy từ chối trả lời.
5. Hãy trả với một giọng điệu thật tự nhiên và thoải mái như thể bạn là một chuyên gia thực sự.

# Định dạng câu trả lời:
1. Câu trả lời của bạn phải thật tự nhiên và không chứa các từ sau: prompt templates, <QUES>, <INS>, <REF>.
2. Câu trả lời của bạn không cần chứa câu hỏi mà người người cung cấp.

Dưới đây là thông tin tôi cung cấp cho bạn: <INS>

<QUES>={query_str} 

<REF>={context_str}
"""

prompt_ = PromptTemplate(prompt_)

# from llama_index.llms.openai import OpenAI

# # resp = OpenAI().complete(final_prompt)

# context = "\n\n".join([node.get_content() for node in final_nodes])
# response = OpenAI().complete(
#     prompt_.format(query_str=query, context_str=context)
# )


# print(response)
