# main.py
import openai
import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from openai import AsyncAzureOpenAI
from app.models.chat_models import ChatMessage


# (lembre-se de ter instalado azure-ai-openai e aiohttp nos requirements)

# --- Logging básico ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurações via Pydantic Settings ---
class AppSettings(BaseSettings):
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str
    AZURE_SEARCH_INDEX_NAME: str

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_MODEL: str = "embedding-deploy"  # ou o nome exato do seu deployment

    AZURE_SEARCH_CHUNK_FIELD: str = "chunk"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = AppSettings()

# --- Modelos Pydantic de Request/Response ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


# --- Dependências (injeção de clientes assíncronos) ---
def get_search_client() -> AsyncSearchClient:
    return AsyncSearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
    )

def get_openai_client() -> OpenAIClient:
    return OpenAIClient(
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_OPENAI_KEY),
        api_version=settings.AZURE_OPENAI_API_VERSION
    )


# --- Cria a aplicação FastAPI ---
app = FastAPI(title="RAG Backend")


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Service is healthy"}


@app.post("/query", response_model=QueryResponse)
async def rag_query(
    req: QueryRequest,
    search_client: AsyncSearchClient = Depends(get_search_client),
    openai_client: OpenAIClient   = Depends(get_openai_client)
):
    logger.info(f"Received query: '{req.query}' with top_k={req.top_k}")

    try:
        # 1) busca simples no Azure Search (sem semantic)
        results = await search_client.search(
            search_text=req.query,
            top=req.top_k
        )

        # 2) coleta o(s) campo(s) de “chunk” retornado(s)
        snippets: List[str] = []
        async for result in results:
            # result pode ser um dict-like ou um objeto; usamos .get() primeiro
            chunk_text = None
            try:
                chunk_text = result.get(settings.AZURE_SEARCH_CHUNK_FIELD)
            except Exception:
                chunk_text = getattr(result, settings.AZURE_SEARCH_CHUNK_FIELD, None)

            if chunk_text:
                snippets.append(chunk_text)
            else:
                logger.warning(f"Campo '{settings.AZURE_SEARCH_CHUNK_FIELD}' não encontrado em: {result}")

        if not snippets:
            prompt_context = "Nenhum contexto específico encontrado."
        else:
            prompt_context = "\n\n---\n\n".join(snippets)

        # 3) montar o prompt para o OpenAI
        system_message = (
            "Você é um assistente de IA prestativo. "
            "Responda à pergunta do usuário com base no contexto fornecido. "
            "Se o contexto não for suficiente, informe educadamente."
        )
        user_prompt = (
            "Use os seguintes trechos de contexto para responder à pergunta.\n\n"
            f"Contexto:\n{prompt_context}\n\n"
            f"Pergunta: {req.query}\n\nResposta:"
        )

        logger.info(f"Enviando prompt para o modelo: {settings.AZURE_OPENAI_MODEL}")

        # 4) chama o Azure OpenAI (chat completions)
        chat = openai_client.chat_completions
        chat_response = await chat.create(
            model=settings.AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=800
        )

        answer = chat_response.choices[0].message.content.strip()
        logger.info(f"Resposta do OpenAI: '{answer}'")

        return QueryResponse(answer=answer, sources=list(set(snippets)))

    except HTTPException:
        raise  # deixa o FastAPI cuidar de erros HTTP “normais” (401, 404, etc)
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ocorreu um erro interno ao processar sua solicitação: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
