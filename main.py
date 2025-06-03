# main.py
import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.identity.aio import DefaultAzureCredential
from openai import AsyncAzureOpenAI, get_bearer_token_provider

# --------------------------------------------------------------------------- #
# 1. Configurações via Pydantic Settings                                      #
# --------------------------------------------------------------------------- #

class AppSettings(BaseSettings):
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str
    AZURE_SEARCH_INDEX_NAME: str

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_MODEL: str = "embedding-deploy"

    AZURE_SEARCH_CHUNK_FIELD: str = "chunk"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = AppSettings()

# --------------------------------------------------------------------------- #
# 2. Modelos de request / response                                            #
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --------------------------------------------------------------------------- #
# 3. Dependências (injeção de clientes)                                       #
# --------------------------------------------------------------------------- #

def get_search_client() -> AsyncSearchClient:
    return AsyncSearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY),
    )


def get_openai_client() -> AsyncAzureOpenAI:
    """
    Cria um cliente AsyncAzureOpenAI autenticado via Managed-Identity / Azure AD.
    O token é obtido on-demand pelo provider.
    """
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, scope="https://cognitiveservices.azure.com/.default"
    )

    return AsyncAzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )


# --------------------------------------------------------------------------- #
# 4. Instância FastAPI                                                        #
# --------------------------------------------------------------------------- #

app = FastAPI(title="RAG Backend")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def rag_query(
    req: QueryRequest,
    search_client: AsyncSearchClient = Depends(get_search_client),
    openai_client: AsyncAzureOpenAI = Depends(get_openai_client),
):
    """
    Executa uma consulta estilo Retrieval-Augmented Generation (RAG):
    1. Busca documentos no Azure AI Search
    2. Monta prompt com os trechos retornados
    3. Gera resposta com Azure OpenAI
    """
    logging.info("Query recebida: %s | top_k=%d", req.query, req.top_k)

    try:
        # ------------------------------------------------------------------- #
        # 4.1 Retrieval – Azure Search                                        #
        # ------------------------------------------------------------------- #
        results = await search_client.search(search_text=req.query, top=req.top_k)

        snippets: List[str] = []
        async for r in results:
            chunk = (
                r.get(settings.AZURE_SEARCH_CHUNK_FIELD)  # dict-like
                if isinstance(r, dict)
                else getattr(r, settings.AZURE_SEARCH_CHUNK_FIELD, None)
            )
            if chunk:
                snippets.append(chunk)

        prompt_context = (
            "\n\n---\n\n".join(snippets) if snippets else "Nenhum contexto encontrado."
        )

        # ------------------------------------------------------------------- #
        # 4.2 Geração – Azure OpenAI                                          #
        # ------------------------------------------------------------------- #
        system_message = (
            "Você é um assistente de IA prestativo. "
            "Responda à pergunta do usuário usando o contexto fornecido. "
            "Se o contexto não for suficiente, diga isso educadamente."
        )

        user_prompt = (
            f"Use os trechos de contexto a seguir para responder.\n\n"
            f"Contexto:\n{prompt_context}\n\n"
            f"Pergunta: {req.query}\n\nResposta:"
        )

        chat_response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
        )

        answer = chat_response.choices[0].message.content.strip()

        return QueryResponse(answer=answer, sources=list(set(snippets)))

    except HTTPException:
        raise  # deixa FastAPI propagar erros HTTP esperados
    except Exception as exc:
        logging.exception("Erro interno processando RAG: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Ocorreu um erro interno ao processar sua solicitação: {exc}",
        )


# --------------------------------------------------------------------------- #
# 5. Execução local (opcional)                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
