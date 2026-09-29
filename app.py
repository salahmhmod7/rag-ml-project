import streamlit as st
import torch
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# إعداد واجهة Streamlit
st.set_page_config(page_title="ML RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Machine Learning RAG Assistant")

# تحميل الموديلات مرة واحدة فقط (Caching)
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="ml_books_rag")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    
    return embeddings, collection, reranker

embeddings_model, collection, reranker = load_rag_pipeline()

# خانة إدخال السؤال
user_query = st.text_input("أسأل عن أي مفهوم في Machine Learning:", placeholder="e.g., What is backpropagation?")

if st.button("بحث وتوليد الإجابة"):
    if user_query.strip():
        with st.spinner("جاري جلب المصادر وتوليد الإجابة..."):
            # 1. Retrieval
            query_embedding = embeddings_model.embed_query(user_query)
            results = collection.query(query_embeddings=[query_embedding], n_results=15)
            
            # 2. Re-ranking
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            pairs = [[user_query, doc] for doc in documents]
            scores = reranker.predict(pairs)
            
            reranked = sorted(
                zip(scores, documents, metadatas),
                key=lambda x: x[0],
                reverse=True
            )[:3]
            
            # 3. Prompting & LLM
            context_str = "\n\n".join([
                f"[Source {i+1}] Book: {meta.get('doc_title')}, Page: {meta.get('start_page')}\nContent: {doc}"
                for i, (score, doc, meta) in enumerate(reranked)
            ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer strictly using the provided context with [Source X] citations.\nContext:\n{context}"),
                ("human", "{question}")
            ])
            
            llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)
            chain = prompt | llm
            response = chain.invoke({"question": user_query, "context": context_str})
            
            # عرض الإجابة
            st.markdown("### 💡 الإجابة:")
            st.write(response.content)
            
            # عرض المصادر
            st.markdown("---")
            st.markdown("### 📖 المصادر المستخدمة:")
            for i, (score, doc, meta) in enumerate(reranked, 1):
                st.markdown(f"**[Source {i}]** {meta.get('doc_title')} (Page {meta.get('start_page')}) - *Score: {score:.4f}*")
    else:
        st.warning("يرجى كتابة سؤال أولاً.")