# ==============================
# IMPORTAÇÕES
# ==============================
import os
import re
import json
import hashlib
# Manipulação de arquivos, JSON e hash (detecção de mudanças)
import unicodedata
import numpy as np
# Texto sem acentos (busca por palavras) e cálculo da reordenação
import faiss
# Biblioteca para busca vetorial eficiente (Facebook AI)
from sentence_transformers import SentenceTransformer
# Modelo para gerar embeddings de texto
from pypdf import PdfReader
# Leitura de arquivos PDF
# ==============================
# CONFIGURAÇÕES DE CAMINHOS
# ==============================

DATA_PATH = "data/cafe_conhecimento.txt"
# Arquivo TXT com base de conhecimento

DOCS_FOLDER = "documentos"
# Pasta contendo PDFs

INDEX_PATH = "rag/index.faiss"
# Caminho onde o índice vetorial será salvo

DOCS_PATH = "rag/docs.json"
# Caminho onde os blocos são salvos, cada um com texto, fonte e página.
# JSON (e não pickle/npy): ler JSON nunca executa código.

META_PATH = "rag/index_meta.json"
# Guarda o fingerprint das fontes usadas para criar o índice

MODEL_NAME = "intfloat/multilingual-e5-small"
# Modelo de embeddings multilíngue (bom em português, aceita até 512 tokens)

PREFIXO_DOC = "passage: "
PREFIXO_PERGUNTA = "query: "
# O E5 foi treinado com esses prefixos: sem eles a qualidade da busca cai

TAMANHO_BLOCO = 800
SOBREPOSICAO = 150
VERSAO_DIVISAO = 3
# Parâmetros de divisão dos textos em blocos.
# VERSAO_DIVISAO: aumente ao mudar o algoritmo de corte, para forçar recriar o índice

# ==============================
# MODELO DE EMBEDDINGS
# ==============================

# Modelo leve e eficiente para gerar vetores semânticos
# Cada texto → vetor numérico que representa significado
model = SentenceTransformer(MODEL_NAME)


# ==============================
# FUNÇÃO: DIVIDIR TEXTO EM BLOCOS
# ==============================

# Níveis de corte, do mais "natural" para o mais grosseiro.
# Todos são zero-width: mantêm o texto original intacto (só separam).
_NIVEIS_DE_CORTE = [
    r"(?<=\n\n)",             # parágrafo
    r"(?<=[.!?])(?=\s)",      # fim de frase
    r"(?<=\n)",               # quebra de linha (PDFs quebram linha no meio da frase)
    r"(?<=\s)",               # palavra
]


def _quebrar_em_segmentos(texto, tamanho, nivel=0):
    """
    Divide o texto em pedaços de no máximo `tamanho` caracteres,
    cortando no nível mais natural possível (parágrafo, frase, linha, palavra).
    Só corta no meio de uma palavra se ela sozinha for maior que o bloco.
    """
    if len(texto) <= tamanho:
        return [texto]

    # Acabaram os níveis: corte seco
    if nivel >= len(_NIVEIS_DE_CORTE):
        return [texto[i:i + tamanho] for i in range(0, len(texto), tamanho)]

    partes = [p for p in re.split(_NIVEIS_DE_CORTE[nivel], texto) if p]
    if len(partes) == 1:
        # Este nível não separou nada: tenta o próximo
        return _quebrar_em_segmentos(texto, tamanho, nivel + 1)

    segmentos = []
    for parte in partes:
        segmentos.extend(_quebrar_em_segmentos(parte, tamanho, nivel + 1))
    return segmentos


def dividir_em_blocos(texto, tamanho=TAMANHO_BLOCO, sobreposicao=SOBREPOSICAO):
    """
    Divide texto em blocos menores (chunks) com sobreposição.

    Por quê?
    - Modelos têm limite de tokens
    - Evita cortar contexto importante
    - Melhora qualidade da busca (RAG)

    Parâmetros:
    - tamanho: tamanho de cada bloco
    - sobreposicao: quantidade de texto repetido entre blocos
    """

    # 1) Quebra o texto em "segmentos" que cabem em um bloco,
    #    preferindo cortar em parágrafo > frase > linha > palavra
    segmentos = _quebrar_em_segmentos(texto, tamanho)

    # 2) Junta segmentos até encher o bloco. Ao fechar um bloco, o próximo
    #    começa repetindo os últimos segmentos (a sobreposição)
    blocos = []
    atual = []          # segmentos do bloco em construção
    tamanho_atual = 0

    for seg in segmentos:
        if atual and tamanho_atual + len(seg) > tamanho:
            bloco = "".join(atual).strip()
            if bloco:
                blocos.append(bloco)

            # Sobreposição: reaproveita os últimos segmentos do bloco fechado
            # (até `sobreposicao` caracteres, sempre deixando espaço para `seg`)
            reaproveitados, soma = [], 0
            for anterior in reversed(atual):
                if soma + len(anterior) > sobreposicao or soma + len(anterior) + len(seg) > tamanho:
                    break
                reaproveitados.insert(0, anterior)
                soma += len(anterior)
            atual, tamanho_atual = reaproveitados, soma

        atual.append(seg)
        tamanho_atual += len(seg)

    bloco = "".join(atual).strip()
    if bloco:
        blocos.append(bloco)

    return blocos


# ==============================
# FUNÇÃO: EXTRAIR TEXTO DE PDF
# ==============================
def extrair_paginas_pdf(caminho):
    """
    Lê um arquivo PDF e devolve uma lista de (número_da_página, texto).
    O número é a posição da página no arquivo PDF (a 1ª página é 1).
    """
    reader = PdfReader(caminho)
    paginas = []
    for numero, pagina in enumerate(reader.pages, start=1):
        # extract_text pode devolver None/vazio (página só com imagem)
        conteudo = pagina.extract_text()
        if conteudo and conteudo.strip():
            paginas.append((numero, conteudo))
    return paginas


def nome_da_fonte(caminho):
    """
    Nome amigável para citar: nome do arquivo sem extensão e sem o
    sufixo " (1)" que o Windows/navegador coloca em downloads repetidos.
    """
    nome = os.path.splitext(os.path.basename(caminho))[0]
    return re.sub(r"\s*\(\d+\)$", "", nome).strip()

# ==============================
# CARREGAMENTO DE DOCUMENTOS TXT
# ==============================
def load_txt_documents():
    """
    Carrega documentos de um arquivo TXT
    e divide em blocos.
    """
    docs = []
    # Verifica se o arquivo existe
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            texto = f.read()
            # Divide o texto em chunks (o TXT não tem páginas: pagina = None)
            for bloco in dividir_em_blocos(texto):
                docs.append({
                    "texto": bloco,
                    "fonte": "Base de conhecimento interna",
                    "pagina": None,
                })
    return docs

# ==============================
# CARREGAMENTO DE DOCUMENTOS PDF
# ==============================
def load_pdf_documents():
    """
    Lê todos os PDFs da pasta e transforma em blocos de texto.
    """
    docs = []
    # Se a pasta não existir, retorna vazio
    if not os.path.exists(DOCS_FOLDER):
        return docs
    # Percorre todos os arquivos da pasta
    for arquivo in sorted(os.listdir(DOCS_FOLDER)):
        # Filtra apenas PDFs
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(DOCS_FOLDER, arquivo)
            print(f"Carregando PDF: {arquivo}")
            fonte = nome_da_fonte(caminho)
            # Cada página é dividida separadamente: assim todo bloco
            # sabe exatamente de qual página veio (necessário para citar)
            for numero, texto in extrair_paginas_pdf(caminho):
                for bloco in dividir_em_blocos(texto):
                    docs.append({"texto": bloco, "fonte": fonte, "pagina": numero})
    return docs


# ==============================
# DETECÇÃO DE MUDANÇAS NAS FONTES
# ==============================

def listar_fontes():
    """
    Lista os arquivos que alimentam o índice (TXT + PDFs), em ordem fixa.
    """
    fontes = []
    if os.path.exists(DATA_PATH):
        fontes.append(DATA_PATH)
    if os.path.exists(DOCS_FOLDER):
        for arquivo in sorted(os.listdir(DOCS_FOLDER)):
            if arquivo.endswith(".pdf"):
                fontes.append(os.path.join(DOCS_FOLDER, arquivo))
    return fontes


def calcular_fingerprint():
    """
    Gera um hash do conteúdo das fontes + parâmetros de divisão + modelo.
    Se qualquer um mudar, o índice salvo deixa de valer.
    """
    h = hashlib.sha256()
    h.update(f"{MODEL_NAME}|{TAMANHO_BLOCO}|{SOBREPOSICAO}|{VERSAO_DIVISAO}".encode())
    for caminho in listar_fontes():
        h.update(caminho.encode())
        with open(caminho, "rb") as f:
            for parte in iter(lambda: f.read(1024 * 1024), b""):
                h.update(parte)
    return h.hexdigest()


def indice_esta_atualizado():
    """
    Compara o fingerprint salvo com o das fontes atuais.
    Sem arquivo de metadados (índice antigo) → considera desatualizado.
    """
    if not os.path.exists(META_PATH):
        return False
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            salvo = json.load(f).get("fingerprint")
    except (json.JSONDecodeError, OSError):
        return False
    return salvo == calcular_fingerprint()


# ==============================
# CRIAÇÃO DO VECTORSTORE (FAISS)
# ==============================

def create_vectorstore():
    """
    Cria ou carrega um índice vetorial FAISS.

    Fluxo:
    1. Verifica se já existe índice salvo
    2. Se sim → carrega
    3. Se não → cria novo índice
    """

    # ==============================
    # CARREGAMENTO DO ÍNDICE EXISTENTE
    # ==============================

    if (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)
            and indice_esta_atualizado()):

        print("Carregando índice FAISS salvo...")

        # Carrega índice FAISS
        index = faiss.read_index(INDEX_PATH)

        # Carrega documentos salvos
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)

        return index, docs

    # ==============================
    # CRIAÇÃO DE NOVO ÍNDICE
    # ==============================
    print("Criando novo índice vetorial (índice ausente ou documentos alterados)...")
    docs = []
    # Carrega documentos TXT
    docs.extend(load_txt_documents())
    # Carrega documentos PDF
    docs.extend(load_pdf_documents())
    # Se não houver documentos, lança erro
    if not docs:
        raise ValueError("Nenhum documento encontrado para indexação.")
    print("Total de blocos carregados:", len(docs))
    # ==============================
    # GERAÇÃO DE EMBEDDINGS
    # ==============================
    # Converte cada bloco em vetor numérico
    embeddings = model.encode(
        [PREFIXO_DOC + d["texto"] for d in docs],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    # Dimensão dos vetores (ex: 384)
    dimension = embeddings.shape[1]


    # ==============================
    # CRIAÇÃO DO ÍNDICE FAISS
    # ==============================
    # IndexFlatL2 → usa distância euclidiana (L2)
    index = faiss.IndexFlatL2(dimension)
    # Adiciona os embeddings ao índice
    index.add(embeddings)
    # ==============================
    # SALVAMENTO EM DISCO
    # ==============================
    # Salva o índice
    faiss.write_index(index, INDEX_PATH)
    # Salva os documentos
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    # Salva a "impressão digital" das fontes para detectar mudanças depois
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": calcular_fingerprint()}, f)
    print("Índice FAISS criado e salvo com sucesso.")
    return index, docs


# ==============================
# APOIO À BUSCA POR PALAVRAS-CHAVE
# ==============================
POOL_BUSCA = 20
# Quantos candidatos a busca vetorial traz antes de reordenar

PESO_LEXICAL = 0.25
# Peso das palavras-chave na reordenação (0 = só vetorial).
# Medido em 20 perguntas: 0,2 a 0,3 acerta 19; 0,5 piora as perguntas sobre os PDFs,
# porque palavras em português favorecem o TXT.

_STOPWORDS = set(
    "o a os as um uma de do da dos das em no na nos nas e ou que q qual quais quem "
    "como para por pq com se ao aos eh ser sao foi sobre entre segundo".split()
)


def _tokens(texto):
    """Palavras úteis do texto: minúsculas, sem acento, sem palavras vazias."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    return [t for t in re.findall(r"[a-z0-9+]+", sem_acento)
            if t not in _STOPWORDS and len(t) > 1]


def _pontuacao_lexical(texto, palavras_pergunta):
    """Quantas vezes as palavras da pergunta aparecem no bloco (suavizado pelo tamanho)."""
    palavras = _tokens(texto)
    if not palavras:
        return 0.0
    return sum(1 for p in palavras if p in palavras_pergunta) / (len(palavras) ** 0.5)


# ==============================
# FUNÇÃO DE BUSCA (RETRIEVAL)
# ==============================
def retrieve(query, index, docs, top_k=4):
    """
    Retorna os blocos mais relevantes para uma consulta.
    Cada bloco é um dict: {"texto": ..., "fonte": ..., "pagina": ... ou None}.
    Parâmetros:
    - query: pergunta do usuário
    - index: índice FAISS
    - docs: lista de blocos (dicts)
    - top_k: quantidade de resultados
    """

    # ==============================
    # EMBEDDING DA CONSULTA
    # ==============================

    query_embedding = model.encode(
        [PREFIXO_PERGUNTA + query],   # lista com uma única pergunta
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    # ==============================
    # BUSCA NO FAISS (mais candidatos do que o necessário)
    # ==============================
    # distances → quão "longe" cada resultado está
    # indices → índices dos documentos mais próximos, do melhor para o pior
    distances, indices = index.search(query_embedding, max(top_k, POOL_BUSCA))

    # O FAISS devolve -1 quando há menos vetores que o pedido
    candidatos = [int(i) for i in indices[0] if 0 <= i < len(docs)]

    # ==============================
    # REORDENA COM PALAVRAS-CHAVE (busca híbrida)
    # ==============================
    # A busca vetorial entende o sentido, mas erra com perguntas informais
    # ("o q e monolito"). Uma pontuação por palavras em comum corrige isso.
    palavras_pergunta = set(_tokens(query))
    if palavras_pergunta and PESO_LEXICAL > 0:
        lexical = np.array([_pontuacao_lexical(docs[i]["texto"], palavras_pergunta) for i in candidatos])
        if lexical.max() > 0:
            lexical = lexical / lexical.max()
        # posição na busca vetorial, de 1 (melhor) a perto de 0 (pior)
        vetorial = np.array([1 - r / len(candidatos) for r in range(len(candidatos))])
        pontos = (1 - PESO_LEXICAL) * vetorial + PESO_LEXICAL * lexical
        candidatos = [candidatos[j] for j in np.argsort(-pontos)]

    return [docs[i] for i in candidatos[:top_k]]