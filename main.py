# main.py
import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.ai.openai import OpenAIClient

# --- modelos de request/response ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- leitura de config pelas env vars ---
SEARCH_ENDPOINT    = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY         = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME  = os.getenv("AZURE_SEARCH_INDEX_NAME")
OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY         = os.getenv("AZURE_OPENAI_KEY")
OPENAI_MODEL       = os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")

if not all([SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX_NAME, OPENAI_ENDPOINT, OPENAI_KEY]):
    raise RuntimeError("Faltam variáveis de ambiente de Search/OpenAI")

# --- instanciação dos clients ---
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)

openai_client = OpenAIClient(
    endpoint=OPENAI_ENDPOINT,
    credential=AzureKeyCredential(OPENAI_KEY)
)

app = FastAPI(title="RAG Backend")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest):
    # 1) busca semântica no Azure Search
    results = search_client.search(
        search_text=req.query,
        query_type="semantic",
        semantic_configuration_name="default",   # ajuste para o nome da sua config
        top=req.top_k,
        answers="extractive"
    )
    # 2) coleto trechos encontrados
    snippets = []
    if results.get_answers():
        # se usou respostas extrativas do Search
        for ans in results.get_answers():
            snippets.append(ans.text)
    else:
        # se não há answers, uso chunks
        for r in results:
            snippets.append(r["chunk"])  # ou o nome do campo no seu índice

    # 3) monta o prompt para o OpenAI
    prompt = (
        "Você é um assistente. Use os seguintes trechos de contexto para responder à pergunta.\n\n"
        + "\n\n---\n\n".join(snippets)
        + f"\n\nPergunta: {req.query}\nResposta:"
    )

    # 4) chama o OpenAI ChatCompletion
    chat = openai_client.chat_completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system", "content":"Você é um assistente prestativo."},
            {"role":"user",   "content":prompt}
        ]
    )
    answer = chat.choices[0].message.content.strip()

    return QueryResponse(answer=answer, sources=snippets)

# opcional, se quiser rodar com 'python main.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
