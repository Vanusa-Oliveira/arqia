# ==============================
# IMPORTAÇÕES
# ==============================

# Flask: framework web para criar a aplicação
# render_template: renderiza páginas HTML
# request: captura dados enviados pelo cliente (frontend)
# Response: permite retornar respostas customizadas (como streaming)
from flask import Flask, render_template, request, Response

# Função customizada que cria o vectorstore (RAG)
from rag.vectorstore import create_vectorstore

import os
import time
import threading
from collections import deque

#CORREÇÃO 2: usa o retrieve do vectorsotre
from rag.vectorstore import retrieve

#CORREÇÃO 3: usar o gera_resposta do llm.py
from rag.llm import gerar_resposta_stream, formatar_contexto, montar_bloco_fontes, MARCADOR_FONTES

# ==============================
# INICIALIZAÇÃO DO FLASK
# ==============================

# Cria a aplicação Flask
app = Flask(__name__)


# ==============================
# CARREGAMENTO DO VECTORSTORE (RAG)
# ==============================

# Aqui acontece:
# - Leitura dos documentos (PDFs, textos, etc.)
# - Geração de embeddings
# - Criação de um índice vetorial (ex: FAISS)

# index → estrutura para busca semântica
# docs → lista original dos documentos
index, docs = create_vectorstore()

# Log para confirmar carregamento
print("Vectorstore carregado com", len(docs), "documentos")


# ==============================
# LIMITE DE PERGUNTAS POR MINUTO
# ==============================

# O plano gratuito da Groq permite ~8.000 tokens por minuto na conta inteira.
# Cada pergunta gasta ~1.400 a 1.900 tokens (contexto + resposta), ou seja,
# cabem cerca de 4 a 5 perguntas por minuto. Os padrões abaixo respeitam isso
# e podem ser mudados no .env (LIMITE_POR_IP e LIMITE_GLOBAL).
JANELA_SEGUNDOS = 60
LIMITE_POR_IP = int(os.getenv("LIMITE_POR_IP", "4"))     # por usuário (endereço IP)
LIMITE_GLOBAL = int(os.getenv("LIMITE_GLOBAL", "5"))     # todos os usuários somados

_trava = threading.Lock()          # o Flask atende várias requisições ao mesmo tempo
_pedidos_por_ip = {}               # ip -> horários dos últimos pedidos
_pedidos_global = deque()          # horários dos últimos pedidos de todos


def _descartar_antigos(fila, agora):
    """Remove da fila os pedidos que já saíram da janela de 1 minuto."""
    while fila and agora - fila[0] >= JANELA_SEGUNDOS:
        fila.popleft()


def verificar_limite(ip):
    """
    Registra um pedido e devolve 0 se está liberado.
    Se o limite estourou, devolve quantos segundos faltam para poder perguntar de novo
    (e o pedido NÃO é registrado).
    """
    agora = time.monotonic()
    with _trava:
        _descartar_antigos(_pedidos_global, agora)
        fila_ip = _pedidos_por_ip.setdefault(ip, deque())
        _descartar_antigos(fila_ip, agora)

        for fila, limite in ((fila_ip, LIMITE_POR_IP), (_pedidos_global, LIMITE_GLOBAL)):
            if len(fila) >= limite:
                return max(1, int(JANELA_SEGUNDOS - (agora - fila[0])) + 1)

        fila_ip.append(agora)
        _pedidos_global.append(agora)

        # Evita crescer sem parar: esquece IPs que ficaram inativos
        if len(_pedidos_por_ip) > 1000:
            for chave in [k for k, f in _pedidos_por_ip.items() if not f or agora - f[-1] >= JANELA_SEGUNDOS]:
                del _pedidos_por_ip[chave]
    return 0


# ==============================
# ROTAS DE PÁGINAS (FRONTEND)
# ==============================

# Página inicial
@app.route("/")
def home():
    # Renderiza o arquivo templates/index.html
    return render_template("index.html")


# Página institucional
@app.route("/institucional")
def institucional():
    # Renderiza o arquivo templates/sobre.html
    return render_template("sobre.html")


# ==============================
# ROTA DA IA (RAG + OLLAMA)
# ==============================

# Essa rota recebe perguntas do frontend e retorna respostas da IA
@app.route("/ask", methods=["POST"])
def ask():
    # ==============================
    # RECEBE A PERGUNTA DO USUÁRIO
    # ==============================
    # Captura o JSON enviado pelo frontend
    # silent=True: JSON inválido vira None em vez de erro 400/415 automático
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return Response("Requisição inválida.", status=400, mimetype="text/plain")
    # Extrai a pergunta e valida
    question = str(data.get("question") or "").strip()
    if not question:
        return Response("Digite uma pergunta.", status=400, mimetype="text/plain")
    if len(question) > 2000:
        return Response("Pergunta muito longa (máximo de 2000 caracteres).", status=400, mimetype="text/plain")

    # ==============================
    # LIMITE DE PERGUNTAS POR MINUTO
    # ==============================
    # (só chega aqui quem mandou uma pergunta válida, então erros de digitação não contam)
    espera = verificar_limite(request.remote_addr or "desconhecido")
    if espera:
        resposta = Response(
            f"Muitas perguntas em pouco tempo. Aguarde {espera} segundos e tente de novo.",
            status=429, mimetype="text/plain")
        resposta.headers["Retry-After"] = str(espera)
        return resposta

    # ==============================
    # HISTÓRICO DA CONVERSA
    # ==============================
    # Lista de {"role": "user"|"assistant", "content": "..."} enviada pelo frontend
    mensagens = data.get("history")
    if not isinstance(mensagens, list):
        mensagens = []
    historico = []
    for m in mensagens[-6:]:
        if not isinstance(m, dict):
            continue
        autor = "Usuário" if m.get("role") == "user" else "Assistente"
        # A lista "Fontes" acrescentada ao final das respostas não vai para o modelo
        texto = str(m.get("content", "")).split(MARCADOR_FONTES)[0]
        texto = texto[:1000]  # limita o tamanho de cada mensagem
        historico.append(f"{autor}: {texto}")

    # Perguntas de continuação ("e as desvantagens dele?") não têm contexto sozinhas.
    # Por isso a busca usa também a última pergunta do usuário.
    ultimas_do_usuario = [m for m in mensagens if isinstance(m, dict) and m.get("role") == "user"]
    consulta_busca = question
    if ultimas_do_usuario:
        consulta_busca = f"{str(ultimas_do_usuario[-1].get('content', ''))[:300]} {question}"

    # ==============================
    # BUSCA SEMÂNTICA (RAG)
    # ==============================
    try:
        # Busca os 4 documentos mais relevantes no vectorstore
        #CORREÇÃO 1: resultados = index.similarity_search(question, k=4)
        # Cada trecho é um dict: {"texto", "fonte", "pagina"}
        trechos = retrieve(consulta_busca, index, docs, top_k=4)
    except Exception as e:
        # Caso a busca falhe
        print("Erro no vectorstore:", e)
        # Fallback: usa os primeiros documentos
        trechos = docs[:4]
    # Junta os trechos numerados ([1], [2]...) em uma única string, para o modelo poder citá-los
    contexto_relevante = formatar_contexto(trechos)
    # ==============================
    # CONSTRUÇÃO DO PROMPT
    # ==============================

    # Aqui você controla o comportamento do modelo (prompt engineering)
    # e injeta o contexto recuperado (RAG)

    # ==============================
    # FUNÇÃO GERADORA (STREAMING)
    # ==============================

    # Essa função envia a resposta aos poucos (token por token)
    #CORREÇÃO 4: agora estou usando o llm.py
    def gerar():
        resposta = ""
        for pedaco in gerar_resposta_stream(contexto_relevante, question, historico):
            # Alguns modelos citam com 【1】 em vez de [1]: padroniza (são caracteres únicos,
            # então a troca é segura mesmo com a resposta chegando em pedaços)
            pedaco = pedaco.replace("【", "[").replace("】", "]")
            resposta += pedaco
            yield pedaco
        # Ao final, acrescenta a lista das fontes que o modelo citou
        yield montar_bloco_fontes(trechos, resposta)
    return Response(gerar(), mimetype="text/plain")

# ==============================
# EXECUÇÃO DA APLICAÇÃO
# ==============================

# Só executa se o arquivo for rodado diretamente
if __name__ == "__main__":
    # Inicia o servidor Flask
    # debug fica DESLIGADO por padrão (o modo debug expõe um console que executa código).
    # Para ligar em desenvolvimento: defina FLASK_DEBUG=1 no ambiente.
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")