# ArqueIA

Assistente com LLM que responde dúvidas sobre **arquitetura de software** a partir de artigos
clássicos da área, mostrando o arquivo e a página de onde veio cada informação.

> 🚧 **Em desenvolvimento.** É um trabalho acadêmico. Como qualquer assistente baseado em modelo de
> linguagem, ele pode errar ou acrescentar algo que não está nos documentos: confira sempre as fontes.

## Como funciona

Técnica: **RAG** (*Retrieval-Augmented Generation*).

1. **Indexação:** os documentos são divididos em blocos (respeitando parágrafos e frases) e
   transformados em vetores com `sentence-transformers` (`intfloat/multilingual-e5-small`).
   O índice fica no **FAISS** e é recriado sozinho quando os documentos mudam.
2. **Busca híbrida:** o FAISS traz 20 candidatos por sentido, e uma pontuação por palavras-chave
   reordena. Os 4 melhores trechos seguem para o modelo.
3. **Resposta:** o modelo (`openai/gpt-oss-120b`, pela API da Groq) é instruído a responder com base
   nos trechos e a citá-los como `[1]`, `[2]`. A lista **Fontes** (arquivo e página) é montada pelo código,
   só com o que o modelo de fato citou.
4. **Interface:** app web em **Flask**, com resposta em streaming, memória da conversa, markdown
   seguro (marked + DOMPurify), botões de copiar e parar, e limite de perguntas por minuto.

Limitações conhecidas: o prompt pede para usar só os documentos, mas isso não é uma garantia; se o
modelo não citar um trecho, a resposta sai sem fonte; a página citada é a posição no PDF, não a
numeração impressa no artigo.

## Como rodar

Requisitos: Python 3.10+ e uma chave da [Groq](https://console.groq.com) (há plano gratuito).

```bash
git clone <url-do-repositorio>
cd <pasta>
python -m venv venv
venv\Scripts\activate        # Windows (cmd). No PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env       # depois edite o .env e coloque sua GROQ_API_KEY
python app.py
```

Abra http://127.0.0.1:5000. A primeira subida é lenta (carrega o PyTorch e o modelo de busca; baixa
o modelo de embeddings) e, se houver PDFs, o índice é criado uma vez.

Para a base completa, coloque os PDFs listados em [`documentos/README.md`](documentos/README.md)
na pasta `documentos/`.

## Medir a qualidade da busca

```bash
python avaliar_busca.py            # roda perguntas de teste, sem usar a Groq (não gasta tokens)
python avaliar_busca.py --peso 0   # compara com busca só vetorial
```

## Estrutura

```
app.py                  servidor Flask (rotas, limite de perguntas, histórico)
rag/vectorstore.py      blocos, embeddings, FAISS, busca híbrida
rag/llm.py              prompt, chamada à Groq (streaming), citação de fontes
avaliar_busca.py        avaliação da busca com perguntas de teste
templates/, static/     interface (chat e página "Quem Somos")
data/                   texto introdutório em português
documentos/             PDFs de referência (não versionados)
```

## Referências

Veja [`documentos/README.md`](documentos/README.md).
