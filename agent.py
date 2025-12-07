# Il cervello. Qui si definisce l'agente e si costruisce il retriever RAG. 
import os
import streamlit as st
import langchain
import redis
from langchain_community.cache import RedisCache
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool, AgentType

# Importiamo dai nostri moduli
from config import QA_PROMPT, SYSTEM_MESSAGE
from tools import calcolatrice_tasse

# Configurazione Cache LLM
try:
    from langchain.globals import set_llm_cache
except ImportError:
    def set_llm_cache(cache):
        langchain.llm_cache = cache

def setup_redis_cache():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        if r.ping():
            set_llm_cache(RedisCache(redis_=r))
            print("✅ Redis Cache attivata!")
        else:
            print("⚠️ Redis non risponde.")
    except Exception:
        print("⚠️ Cache disattivata (Redis non trovato).")


def get_agent_executor(vector_db):
    # 1. Modello
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.01,
        num_ctx=4096,
    )

    # 2. Retriever
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12},
    )

    # 3. Funzione RAG Interna
    def rag_qa(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "Non lo so in base ai documenti disponibili."

        q = question.lower()
        
        # Filtro per borse di studio
        if "borsa" in q or "bando" in q:
            scholarship_docs = [d for d in docs if "borsa" in d.metadata.get("source","").lower() or "bando" in d.metadata.get("source","").lower()]
            if scholarship_docs:
                docs = scholarship_docs

        # Logica di ordinamento (Priority)
        def doc_priority(doc):
            fname = os.path.basename(doc.metadata.get("source", "")).lower()
            score = 0
            if "borsa" in fname: score -= 5
            if "bando" in fname: score -= 4
            if "guida" in fname: score += 2
            return score
        
        docs = sorted(docs, key=doc_priority)

        # Costruzione contesto
        context_chunks = []
        snippets = []
        refs = []

        for i, doc in enumerate(docs, start=1):
            text = doc.page_content.strip().replace("\n", " ")
            text_short = text[:350] + "..." if len(text) > 350 else text
            filename = os.path.basename(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", None)

            context_chunks.append(f"[{i}] {text}")
            
            ref_str = f"- {filename}"
            if page is not None:
                ref_str += f" (pag. {page + 1})"
            
            refs.append(ref_str)
            snippets.append(f"📄 *{ref_str}*: {text_short}")

        context = "\n\n".join(context_chunks)
        full_prompt = QA_PROMPT.format(context=context, question=question)
        
        answer = llm.predict(full_prompt)

        if snippets:
            answer += "\n\n---\nEstratti dai documenti utilizzati:\n" + "\n\n".join(snippets)
        if refs:
            answer += "\n\nFonti nei documenti:\n" + "\n".join(refs)

        return answer

    # 4. Tool RAG
    rag_tool = Tool(
        name="KnowledgeBase_Universitaria",
        func=rag_qa,
        description="Utile per rispondere a domande su regolamenti e documenti.",
        return_direct=True,
    )

    tools = [rag_tool, calcolatrice_tasse]

    # 5. Creazione Agente
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM_MESSAGE},
        max_iterations=6,
        early_stopping_method="generate",
    )

    return agent

# Wrapper Cache
@st.cache_resource(show_spinner=False)
def get_cached_agent(_vector_db):
    return get_agent_executor(_vector_db)