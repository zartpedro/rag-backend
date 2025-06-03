import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from azure.identity.aio import DefaultAzureCredential
from azure.identity import get_bearer_token_provider_async
from azure.ai.openai.aio import AsyncAzureOpenAI
from dotenv import load_dotenv

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente de um arquivo .env (para desenvolvimento local)
load_dotenv()

# Configuração da Aplicação FastAPI
app = FastAPI(
    title="Azure RAG FastAPI Application",
    description="Uma API para interagir com um modelo RAG usando Azure OpenAI e Azure AI Search.",
    version="1.0.0"
)

# Variáveis de Ambiente (configure-as no seu Azure App Service)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT") # Ex: gpt-4o-mini [cite: 128]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") # Ex: text-embedding-ada-002 [cite: 54]
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_URL") # Nome da variável conforme doc [cite: 69]
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME") [cite: 69]
# A API version mencionada no documento é "2024-10-21"[cite: 115].
# Se esta for uma versão preview específica necessária, use-a. Caso contrário, uma mais recente estável.
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")


# Validação inicial das variáveis de ambiente
required_env_vars = {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_GPT_DEPLOYMENT": AZURE_OPENAI_GPT_DEPLOYMENT,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
}

missing_vars = [var for var, val in required_env_vars.items() if val is None]
if missing_vars:
    error_message = f"Erro crítico: As seguintes variáveis de ambiente não estão configuradas: {', '.join(missing_vars)}"
    logger.error(error_message)
    # Em um cenário de produção, você pode querer que o app falhe ao iniciar.
    # Para este exemplo, apenas logamos, mas o app provavelmente não funcionará corretamente.
    # raise EnvironmentError(error_message) # Descomente para falhar se variáveis estiverem faltando


# Modelos Pydantic para Request e Response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    # Você pode adicionar mais campos aqui, como conversation_id para gerenciar o histórico

class Citation(BaseModel):
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    filepath: Optional[str] = None
    chunk_id: Optional[str] = None

class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    citations: List[Citation] = []

class ChatResponse(BaseModel):
    message: ChatCompletionMessage


# Inicializar o cliente Azure OpenAI de forma assíncrona
# Usaremos DefaultAzureCredential para autenticação via Identidade Gerenciada no Azure [cite: 115]
try:
    credential = DefaultAzureCredential()
    # O token provider é necessário para AsyncAzureOpenAI com azure_ad_token_provider
    # A scope é para o serviço cognitivo do Azure [cite: 115]
    token_provider = get_bearer_token_provider_async(credential, "https://cognitiveservices.azure.com/.default")

    openai_client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_ad_token_provider=token_provider
    )
except Exception as e:
    logger.error(f"Falha ao inicializar o cliente Azure OpenAI ou credenciais: {e}")
    openai_client = None # Garante que o cliente não seja usado se a inicialização falhar


@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    if not openai_client:
        raise HTTPException(status_code=503, detail="Serviço OpenAI não está disponível devido a erro de configuração.")
    if not all(required_env_vars.values()): # Verifica novamente se todas as vars estão carregadas
         raise HTTPException(status_code=500, detail=f"Configuração incompleta do servidor. Variáveis faltando: {', '.join(missing_vars)}")


    # Pegar a última mensagem do usuário para a consulta
    # Para um chat real, você deveria enviar um histórico mais completo das mensagens.
    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem de usuário encontrada no request.")

    # Montar a lista de mensagens para a API. Idealmente, incluiria o histórico da conversa.
    # Por simplicidade, este exemplo apenas pega a última mensagem do usuário.
    # Para um sistema de chat completo, você precisaria gerenciar e passar o histórico da conversa.
    # messages_for_api = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # A API "on your data" geralmente funciona melhor com a pergunta do usuário como a última mensagem
    # e o sistema pode precisar de um prompt de sistema.
    # O exemplo do tutorial não detalha a construção de `messages`,
    # mas foca no `extra_body` com `data_sources`.
    # Assumindo que a última mensagem é a pergunta principal.
    messages_for_api = [
        # {"role": "system", "content": "Você é um assistente de IA útil que responde perguntas com base nos documentos fornecidos."},
        {"role": "user", "content": user_message}
    ]


    # Configurar a fonte de dados (Azure AI Search) [cite: 117, 121]
    data_source = {
        "type": "azure_search",
        "parameters": {
            "endpoint": AZURE_SEARCH_ENDPOINT,
            "index_name": AZURE_SEARCH_INDEX_NAME,
            "authentication": {
                "type": "system_assigned_managed_identity" # [cite: 4, 117]
            },
            "query_type": "vector_semantic_hybrid", # [cite: 121]
            "semantic_configuration": f"{AZURE_SEARCH_INDEX_NAME}-semantic-configuration", # [cite: 121, 125]
            "embedding_dependency": { # [cite: 121]
                "type": "deployment_name",
                "deployment_name": AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            },
            # Outros parâmetros opcionais:
            # "top_n_documents": 5,
            # "strictness": 3,
            # "in_scope": True, # Forçar o modelo a usar apenas os dados do índice
            # "role_information": "Você é um assistente de IA que ajuda usuários com informações de documentos."
        }
    }

    try:
        logger.info(f"Enviando requisição para o deployment GPT: {AZURE_OPENAI_GPT_DEPLOYMENT}")
        completion = await openai_client.chat.completions.create(
            model=AZURE_OPENAI_GPT_DEPLOYMENT,
            messages=messages_for_api,
            extra_body={ # Para usar Azure AI Search como fonte de dados [cite: 18, 117]
                "data_sources": [data_source]
            },
            stream=False # O tutorial usa stream=False [cite: 117]
        )

        response_message = completion.choices[0].message

        # Extrair citações do contexto da mensagem [cite: 100, 102]
        citations_data = []
        if response_message.context and response_message.context.get("citations"):
            for cit_data in response_message.context["citations"]:
                citations_data.append(Citation(
                    content=cit_data.get("content"),
                    title=cit_data.get("title"),
                    url=cit_data.get("url"),
                    filepath=cit_data.get("filepath"), # 'filepath' é mais comum que 'filename'
                    chunk_id=cit_data.get("chunk_id")
                ))
        
        # O conteúdo da resposta já pode conter referências como [doc1], [doc2] etc. [cite: 103]
        chat_response_content = response_message.content

        return ChatResponse(
            message=ChatCompletionMessage(
                role="assistant",
                content=chat_response_content,
                citations=citations_data
            )
        )

    except Exception as e:
        logger.error(f"Erro ao chamar a API de Chat Completions: {e}")
        # Verifique se 'e' é uma exceção da API OpenAI e capture detalhes específicos se necessário
        # Por exemplo, e.status_code, e.response.json()
        error_detail = str(e)
        status_code = 500
        if hasattr(e, 'status_code'): # Erros da API OpenAI podem ter status_code
            status_code = e.status_code
            if hasattr(e, 'message'):
                error_detail = e.message
            elif hasattr(e, 'body') and e.body and 'message' in e.body: # Para azure.core.exceptions.HttpResponseError
                 error_detail = e.body['message']
            elif hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_detail = e.response.text


        raise HTTPException(status_code=status_code, detail=f"Erro ao processar a requisição de chat: {error_detail}")

@app.get("/health", summary="Verifica a saúde da aplicação")
async def health_check():
    # Uma verificação de saúde básica. Pode ser expandida para checar a conectividade com os serviços Azure.
    return {"status": "ok", "openai_client_initialized": openai_client is not None}

# Para executar localmente com Uvicorn (para desenvolvimento):
# uvicorn main:app --reload