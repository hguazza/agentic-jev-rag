Aqui está um modelo completo e estruturado de **`PROJECT_CONTEXT.md`** para você colocar na raiz do seu projeto no VSCode. Ele foi formatado com todo o contexto necessário para que o Claude (ou o Cursor) entenda exatamente os objetivos, a arquitetura em 3 níveis, a extração de dados públicos jurídicos via API e a transição de Jupyter Notebooks para uma API em FastAPI.

---

### Conteúdo para o seu `PROJECT_CONTEXT.md`

```markdown
# PROJECT_CONTEXT.md — Legal RAG Evolution: Vanilla ➔ Agentic ➔ Agentic Jev RAG

## 🎯 Objetivo do Projeto
Desenvolver um PoC (Proof of Concept) de alta performance focado no domínio jurídico, demonstrando a evolução de uma arquitetura de RAG (Retrieval-Augmented Generation) em 3 níveis sucessivos:
1. **Level 1: Vanilla RAG** — Pipeline linear clássico de busca e geração.
2. **Level 2: Agentic RAG** — Agente autônomo com loops de decisão, reescrita de queries, busca híbrida e reranking.
3. **Level 3: Agentic Jev RAG** — Agente otimizado com **Jev (Typesafe AI)** atuando como modelo *System 1* para roteamento ultrarrápido, guardrails de segurança (Auto-mode), filtragem paralela de chunks e avaliações em tempo real (*Jev-as-a-Judge*).

O projeto iniciará com **Jupyter Notebooks** para experimentação e validação rápida, sendo posteriormente refatorado para uma aplicação **FastAPI** pronta para produção.

---

## 🏛️ Fonte de Dados Jurídicos (Open API)
Utilizaremos dados públicos jurídicos extraídos via REST API:
- **API Escolhida:** API do *Querido Diário* (Open Knowledge Brasil) ou API de Jurisprudência (STF/STJ) / Dados Abertos do Judiciário.
- **Tipo de Dados:** Diários oficiais municipais/estaduais, ementas de processos, acórdãos e peças processuais públicas em JSON/PDF.
- **Campos Principais de Metadados:** `id`, `tribunal/municipio`, `data_publicacao`, `relator/orgao`, `inteiro_teor`, `categoria_juridica`.

---

## 🏗️ As 3 Camadas da Arquitetura

### Level 1: Vanilla RAG (`01_vanilla_rag.ipynb`)
- **Ingestão:** Fetch dos dados via Open API ➔ Chunking fixo por tamanho de caracteres ➔ Geração de Embeddings (`text-embedding-3-small` ou `cohere-embed`).
- **Armazenamento:** Banco vetorial local (Qdrant em memória / ChromaDB).
- **Retrieval:** Busca por similaridade de cosseno (top-k simples).
- **Generation:** Prompt augmentation direto para o LLM generativo (System 2).

### Level 2: Advanced & Agentic RAG (`02_agentic_rag.ipynb`)
- **Ingestão:** Semantic/Hierarchical Chunking por seções jurídicas + preservação de metadados rígidos.
- **Retrieval Híbrido:** BM25 (busca esparsa por artigos e números de processos) + Embeddings Densos.
- **Pós-Retrieval:** Reranking de chunks com Cross-Encoder (`Cohere Rerank` ou `bge-reranker-large`).
- **Agentic Loop:** Reescrita dinâmica de query, decomposição de sub-perguntas e loops de verificação de resposta via LLM generativo.

### Level 3: Agentic Jev RAG (`03_agentic_jev_rag.ipynb`)
Integração do **Jev** como modelo **System 1** via pacote `langchain-typesafe`:
- **Roteamento Instantâneo (`Choice`):** Decisão imediata sobre qual coleção ou ferramenta consultar, reduzindo a latência de roteamento.
- **Filtragem e Contradição em Paralelo (`Score` & `Nule`):** O Jev recebe múltiplos chunks e avalia em uma única chamada paralela se cada chunk atende à dúvida e se está vigente/sem contradições.
- **Guardrail de Segurança / Auto-mode (`Nule`):** Middleware que classifica o risco da intenção/chamada de ferramenta antes de autorizar a execução (ex: bloqueio de PII ou ações restritas).
- **Jev-as-a-Judge (Online Evals):** Avaliação de *Groundedness* e presença de citações das fontes na resposta final do LLM a uma fração do custo e com baixíssima latência.

---

## 🛠️ Tech Stack & Dependências

- **Linguagem:** Python 3.11+
- **Framework de Agentes:** LangChain / LangGraph
- **Agente System 1:** `langchain-typesafe` (`TypeSafeClassifier` / Jev API)
- **Vector DB:** Qdrant Client (modo local/in-memory ou Docker)
- **Framework Web (Fase Final):** FastAPI + Uvicorn + Pydantic v2
- **Data Fetching:** `httpx` / `requests` para integração com a API Jurídica
- **Reranking:** `cohere` / `sentence-transformers`

---

## 🗺️ Roadmap de Execução

### Fase 1 — Jupyter Notebooks (Prototipagem & Benchmark)
1. `notebooks/01_vanilla_rag.ipynb`:
   - Consumir API jurídica aberta e baixar um lote de acórdãos/diários.
   - Criar o fluxo simples VectorDB ➔ Retrieval ➔ OpenAI LLM.
2. `notebooks/02_agentic_rag.ipynb`:
   - Implementar busca híbrida (BM25 + Dense) e reranker.
   - Adicionar agente com chamadas de ferramentas e reescrita de query.
3. `notebooks/03_agentic_jev_rag.ipynb`:
   - Configurar o `TypeSafeClassifier` com `langchain-typesafe`.
   - Substituir o roteador e os guardrails do LLM comum pelo Jev.
   - Executar benchmark comparativo de latência (ms vs. segundos) e custo entre Level 2 e Level 3.

### Fase 2 — Refatoração para FastAPI
Estruturação de um projeto modular em Python:
```
app/
├── core/
│   ├── config.py
│   └── security.py
├── services/
│   ├── legal_api_service.py     # Ingestão da Open API Jurídica
│   ├── vector_store_service.py  # Gerenciamento de embeddings e Qdrant
│   ├── rag_vanilla.py
│   ├── rag_agentic.py
│   └── rag_jev.py               # Agente otimizado com Jev System 1
├── api/
│   ├── v1/
│   │   ├── endpoints/
│   │   │   ├── ingest.py        # POST /api/v1/ingest
│   │   │   ├── query.py         # POST /api/v1/query (suporta modo vanilla, agentic, jev)
│   │   │   └── eval.py          # POST /api/v1/eval (Jev-as-a-Judge)
```

---

## ⚡ Regras Globais para o Claude / Assistente no VSCode
- Escreva código modular, assíncrono (`async/await`) e fortemente tipado com Pydantic.
- Mantenha comentários claros explicando a diferença entre decisões **System 1 (Jev)** e raciocínio **System 2 (LLM Generativo)**.
- Sempre valide os dados que chegam da Open API Jurídica antes de enviar para o pipeline de embedding.
```

---

### Por que esta estrutura vai impressionar no projeto:

1. **Separação Clara das Camadas:** Demonstra maturidade ao mostrar que o RAG evolui conforme as necessidades de precisão, até chegar na otimização de latência e custos.
2. **Uso Estratégico do Jev:** Destaca que o Jev é um modelo **System 1** focado em decisões tipadas e probabilísticas (*Choice*, *Score*, *Nule*), operando entre **20x e 200x mais rápido** e **40x a 400x mais barato** do que colocar um LLM generativo lento no loop do agente.
3. **Integração Real com `langchain-typesafe`:** O código no Level 3 usará o `TypeSafeClassifier` nativo para roteamento, *Auto-mode guardrails* e avaliações *Jev-as-a-Judge*.

💡 Quando criar a pasta do projeto no VSCode, basta salvar este conteúdo como `PROJECT_CONTEXT.md` ou `CLAUDE.md`. Quer que comecemos gerando o código do primeiro notebook (`01_vanilla_rag.ipynb`) com o consumo da API jurídica?
