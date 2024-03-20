from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
import datetime
import os
import re

os.environ["OPENAI_API_KEY"] = ""

input_file_path = ''
reader = PdfReader(input_file_path)


raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

len(raw_text)

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 3500,
    chunk_overlap  = 50,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
print(len(texts))
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

prompt_template = """ I want you to act as a senior professor, combining your knowledge and the following pieces of context to answer the question.


{context}

Question: {question}
Answer in English:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", prompt=PROMPT)

def query_func(str):
  with get_openai_callback() as cb:
    query = str
    docs = docsearch.similarity_search(query)
    res = chain.run(input_documents=docs, question=query)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    input_file_name = re.search(r'(.+?)(?=_\d{8}_\d{6}\.pdf)', input_file_path).group(1)
    output_file_name = f"{input_file_name}__output_{timestamp}.txt"

    with open(output_file_name, "w", encoding="utf-8") as file:
        file.write(f"Query String: {query}\n")
        file.write(f"Prompt: {prompt_template}\n")  # Save the prompt_template
        file.write(f"Output: {res}\n")


    print(f"Output: {res}")


query_func("Detailed inference of several cutting-edge research directions and avoiding duplicating existing research.")
