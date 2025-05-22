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
        # 1) busca semântica no Azure Search (usando await com cliente assíncrono)
        results = await search_client.search(
            search_text=req.query,
            query_type="semantic",
            semantic_configuration_name=settings.AZURE_SEARCH_SEMANTIC_CONFIG_NAME,
            top=req.top_k,
            query_caption="extractive", # Para respostas extrativas
            query_answer="extractive",
            top_k_answers=3 # Número de respostas extrativas desejadas
        )

        # 2) coleto trechos encontrados
        snippets = []
        # A API de respostas extrativas (answers) é mais robusta
        answer_results_list = await results.get_answers()  # <<-- ADICIONE O AWAIT AQUI
        if answer_results_list:  # Verifique se a lista não está vazia
            for answer_result in answer_results_list:  # Loop FOR normal sobre a lista
                snippets.append(answer_result.text)

        # Se não houver 'answers', podemos usar os 'captions' ou os próprios documentos
        if not snippets:
            logger.info("No direct answers found, collecting captions or document chunks.")
            async for result in results: # Iterar sobre os resultados assíncronos
                if result.captions:
                    for caption in result.captions:
                        snippets.append(caption.text)
                else:
                    # Se não houver captions, pegue o campo de chunk do documento
                    # Assumindo que 'result' é um dicionário ou objeto com o campo
                    try:
                        snippets.append(result[settings.AZURE_SEARCH_CHUNK_FIELD])
                    except KeyError:
                        logger.warning(f"Field '{settings.AZURE_SEARCH_CHUNK_FIELD}' not found in search result.")
                    except TypeError: # Se 'result' não for subscriptable
                         logger.warning(f"Search result of type {type(result)} is not subscriptable for field '{settings.AZURE_SEARCH_CHUNK_FIELD}'.")


        if not snippets:
            logger.warning(f"No snippets found for query: '{req.query}'")
            # Decida o que fazer: retornar um erro, ou tentar responder sem contexto
            # return QueryResponse(answer="Não encontrei informações suficientes para responder.", sources=[])
            # Ou permitir que o LLM tente responder sem contexto específico
            prompt_context = "Nenhum contexto específico encontrado."
        else:
            prompt_context = "\n\n---\n\n".join(snippets)

        # 3) monta o prompt para o OpenAI
        system_message = "Você é um assistente de IA prestativo. Responda à pergunta do usuário com base no contexto fornecido. Se o contexto não for suficiente, informe educadamente."
        user_prompt = (
            "Use os seguintes trechos de contexto para responder à pergunta.\n\n"
            f"Contexto:\n{prompt_context}\n\n"
            f"Pergunta: {req.query}\n\nResposta:"
        )

        logger.info(f"Sending prompt to OpenAI model: {settings.AZURE_OPENAI_MODEL}")

        # 4) chama o OpenAI ChatCompletion (usando await)
        chat_completion = await openai_client.chat_completions.create(
            model=settings.AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800 # Defina um max_tokens apropriado
        )

        answer = chat_completion.choices[0].message.content.strip() if chat_completion.choices else "Não recebi uma resposta do modelo."
        logger.info(f"Received answer from OpenAI: '{answer}'")

        return QueryResponse(answer=answer, sources=list(set(snippets))) # list(set()) para remover duplicatas

    except HTTPException: # Re-throw HTTPExceptions para que o FastAPI as manipule
        raise
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True) # exc_info=True para logar o stack trace
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar sua solicitação: {str(e)}")

# opcional, se quiser rodar com 'python main.py'
if __name__ == "__main__":
    import uvicorn
    # Para rodar localmente, as variáveis de ambiente podem ser carregadas de um .env se você tiver python-dotenv instalado
    # ou configuradas no seu ambiente.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)