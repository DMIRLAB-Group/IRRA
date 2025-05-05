from tqdm import tqdm
from os import path, listdir
from langchain_chroma import Chroma
from uuid import uuid5, NAMESPACE_X500
from langchain_core.documents import Document
from BCEmbedding.tools.langchain import BCERerank
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings import HuggingFaceEmbeddings


class DataBase:
    def __init__(self, chunk_size=512, batch_size=512, top_n=4, k=20, device='cuda:0') -> None:
        self._data_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'database', 'crossner', 'docs')
        self._chroma_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'database', 'crossner', f'chroma-{chunk_size}')
        self._chunklevel_retrievers = {}
        self._docs_count = {}
        self._text_spitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        self._embed_model = HuggingFaceEmbeddings(
            model_name=path.join(path.dirname(path.dirname(path.dirname(__file__))), 'models', 'bce-embedding'),
            model_kwargs={'device': device},
            encode_kwargs={'batch_size': batch_size, 'normalize_embeddings': True}
        )
        self._reranker = BCERerank(**{'model': path.join(path.dirname(path.dirname(path.dirname(__file__))), 'models', 'bce-reranker'), 'top_n': top_n, 'device': device})

        self._init_retrievers(k=k, batch_size=batch_size)

    def _init_retrievers(self, k=20, batch_size=16) -> None:
        for data_file in tqdm(listdir(self._data_dir), total=len(listdir(self._data_dir))):
            if not data_file.endswith('.txt'):
                continue
            data_file_name = data_file.split('.')[0]
            with open(path.join(self._data_dir, data_file), 'r') as f:
                data_file = f.read()
            data_file = data_file.split('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
            docs = []
            for doc in data_file:
                doc = doc.strip()
                doc_id = str(uuid5(NAMESPACE_X500, doc))
                doc = Document(
                    page_content=doc,
                    metadata={'doc_id': doc_id}
                )
                docs.append(doc)
            chunks = self._text_spitter.split_documents(docs)
            persist_directory = path.join(self._chroma_dir, data_file_name)
            if path.exists(persist_directory):
                retriever = Chroma(persist_directory=persist_directory, embedding_function=self._embed_model)
                self._docs_count[data_file_name] = retriever._collection.count()
                retriever = retriever.as_retriever(search_type="similarity", search_kwargs={'k': k})
            else:
                retriever = Chroma(persist_directory=persist_directory, embedding_function=self._embed_model, create_collection_if_not_exists=True)
                for i in tqdm(range(0, len(chunks), 41665), total=len(chunks) // 41665 + 1):
                    retriever.add_documents(chunks[i: i + 41665])
                self._docs_count[data_file_name] = retriever._collection.count()
                retriever = retriever.as_retriever(search_type='similarity', search_kwargs={'k': k})
            self._chunklevel_retrievers[data_file_name] = ContextualCompressionRetriever(
                base_compressor=self._reranker, base_retriever=retriever
            )
    
    def get(self, domain: str, query: str):
        return self._chunklevel_retrievers[domain].get_relevant_documents(query)
    
    def count(self):
        return self._docs_count