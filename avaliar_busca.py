# ==============================
# AVALIAÇÃO DA BUSCA (não usa a Groq, então não gasta tokens)
# ==============================
# Para cada pergunta, confere se o trecho que CONTÉM a resposta aparece
# entre os resultados da busca. Rode sempre que mudar algo na busca
# (top_k, PESO_LEXICAL, modelo de embeddings, tamanho dos blocos...):
#
#     python avaliar_busca.py
#
# Para testar um ajuste sem editar o arquivo:
#     python avaliar_busca.py --peso 0        (só busca vetorial)
#     python avaliar_busca.py --top-k 6

import argparse

import rag.vectorstore as vs

# (pergunta, texto que PRECISA aparecer em algum dos trechos recuperados, sem diferenciar maiúsculas)
CASOS = [
    # Base de conhecimento interna (com perguntas formais e informais)
    ("o q e monolito", "O sistema inteiro (interface"),
    ("o que é monolito", "O sistema inteiro (interface"),
    ("O que é o monolito?", "O sistema inteiro (interface"),
    ("vantagens do monolito", "Mais simples de desenvolver"),
    ("desvantagens monolito", "escala tudo junto"),
    ("o q sao microservicos", "pequenos serviços independentes"),
    ("vantagens e desvantagens dos microserviços?", "escalabilidade cirúrgica"),
    ("o que é arquitetura em camadas", "camadas horizontais"),
    ("camadas n-tier", "camadas horizontais"),
    ("o q é event driven", 'reage a "eventos"'),
    ("o que é débito técnico", "Débito Técnico"),
    ("quais os 3 pilares da arquitetura", "Atributos de Qualidade"),
    # PDFs em inglês, perguntados em português e em inglês
    ("o que é software architecture segundo Perry e Wolf", "rationale"),
    ("explique elementos, forma e racional na arquitetura", "rationale"),
    ("o que é pipes and filters", "pipes"),
    ("visão lógica do modelo 4+1", "logical view"),
    ("what is the 4+1 view model", "logical"),
    ("estilos arquiteturais de Garlan e Shaw", "style"),
    ("o que é a visão de processos", "process view"),
    ("architectural styles pipes filters layered", "layered"),
]


def main():
    parser = argparse.ArgumentParser(description="Mede se a busca encontra o trecho certo.")
    parser.add_argument("--top-k", type=int, default=4, help="trechos recuperados (padrão: 4, igual ao app)")
    parser.add_argument("--peso", type=float, default=None, help="PESO_LEXICAL a testar (0 = só vetorial)")
    args = parser.parse_args()

    if args.peso is not None:
        vs.PESO_LEXICAL = args.peso

    index, docs = vs.create_vectorstore()
    print(f"\nPESO_LEXICAL={vs.PESO_LEXICAL}  POOL_BUSCA={vs.POOL_BUSCA}  top_k={args.top_k}\n")

    acertos = 0
    for pergunta, esperado in CASOS:
        trechos = vs.retrieve(pergunta, index, docs, top_k=args.top_k)
        achou = any(esperado.lower() in t["texto"].lower() for t in trechos)
        acertos += achou
        print(f"  {'ok    ' if achou else 'FALHOU'}  {pergunta}")

    print(f"\n{acertos} de {len(CASOS)} perguntas encontraram o trecho certo "
          f"({100 * acertos // len(CASOS)}%)")


if __name__ == "__main__":
    main()
