# main.py
import os
from typing import List
import logging # Adicionar para logging

from fastapi import FastAPI, HTTPException, Depends # Adicionar Depends para injeção
from pydantic import BaseModel, Field # Adicionar Field para validação/exemplos
from pydantic_settings import BaseSettings, SettingsConfigDict # Para configuração elegante

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncSearchClient # Usar cliente assíncrono
from openai import AsyncAzureOpenAI # Usar cliente assíncrono

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Modelos Pydantic para Configuração (Mais Elegante) ---
class AppSettings(BaseSettings):
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str
    AZURE_SEARCH_INDEX_NAME: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"  # Exemplo, ajuste conforme necessário
    AZURE_OPENAI_MODEL: str = "gpt-35-turbo" # Ou o seu deployment name
    # Opcional: para configurar o nome da configuração semântica do Azure Search
    AZURE_SEARCH_SEMANTIC_CONFIG_NAME: str = "default"
    AZURE_SEARCH_CHUNK_FIELD: str = "chunk" # Campo do seu índice que contém o texto

    # Carrega de .env e variáveis de ambiente do sistema
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = AppSettings() # Carrega as configurações na inicialização

# --- modelos de request/response (como antes, talvez com exemplos) ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20) # Adiciona validação e exemplo

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- Clientes (instanciados de forma assíncrona e gerenciados pelo FastAPI) ---
# Não é ideal instanciar clientes globalmente se eles precisam ser assíncronos
# e gerenciados por eventos de startup/shutdown ou injeção de dependência.

# Usaremos injeção de dependência para os clientes.
# Você pode movê-los para um módulo de 'serviços' se o app crescer.

def get_search_client() -> AsyncSearchClient:
    return AsyncSearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
    )

def get_openai_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=settings.AZURE_OPENAI_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

app = FastAPI(title="RAG Backend")

# --- Endpoints ---
@app.get("/")
async def root():
    return {"status":"ok"}


@app.get("/health")
async def health():
    # Poderia adicionar verificações de dependência aqui (e.g., ping no search/openai)
    return {"status": "ok", "message": "Service is healthy"}

@app.post("/query", response_model=QueryResponse)
async def rag_query(
    req: QueryRequest,
    search_client: AsyncSearchClient = Depends(get_search_client),
    openai_client: AsyncAzureOpenAI = Depends(get_openai_client)
):
    logger.info(f"Received query: '{req.query}' with top_k={req.top_k}")

    try:
        # 1) Busca full-text no Azure Search (sem semântica)
        results = await search_client.search(
            search_text=req.query,
            top=req.top_k  # apenas retorna os 'top_k' primeiros documentos
        )

        # 2) Coleta trechos (chunks) dos documentos retornados
        snippets = []
        async for result in results:
            # Se houver um campo de legendas (captions), você pode coletá-las,
            # mas em busca full-text normalmente pegamos o chunk principal:
            if result.captions:
                for caption in result.captions:
                    snippets.append(caption.text)
            else:
                try:
                    # Ajuste este campo conforme o nome correto do chunk no seu índice
                    snippets.append(result[settings.AZURE_SEARCH_CHUNK_FIELD])
                except (KeyError, TypeError):
                    logger.warning(
                        f"Campo '{settings.AZURE_SEARCH_CHUNK_FIELD}' não encontrado no resultado ou inacessível."
                    )

        # Se não houver snippets, defina contexto genérico
        if not snippets:
            logger.warning(f"No snippets found for query: '{req.query}'")
            prompt_context = "Nenhum contexto específico encontrado."
        else:
            # Junta todos os snippets separados por delimitador
            prompt_context = "\n\n---\n\n".join(snippets)

        # 3) Monta o prompt para o OpenAI
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

        logger.info(f"Sending prompt to OpenAI model: {settings.AZURE_OPENAI_MODEL}")

        # 4) Chama o OpenAI ChatCompletion (assíncrono)
        chat_completion = await openai_client.chat_completions.create(
            model=settings.AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=800  # ajuste conforme necessário
        )

        answer = (
            chat_completion.choices[0].message.content.strip()
            if chat_completion.choices else
            "Não recebi uma resposta do modelo."
        )
        logger.info(f"Received answer from OpenAI: '{answer}'")

        # Retorna as fontes (snippets) sem duplicatas
        return QueryResponse(answer=answer, sources=list(set(snippets)))

    except HTTPException:
        # Se já for um HTTPException, apenas re-lança para o FastAPI tratar
        raise
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ocorreu um erro interno ao processar sua solicitação: {str(e)}"
        )


# opcional, se quiser rodar com 'python main.py'
if __name__ == "__main__":
    import uvicorn
    # Para rodar localmente, as variáveis de ambiente podem ser carregadas de um .env se você tiver python-dotenv instalado
    # ou configuradas no seu ambiente.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)