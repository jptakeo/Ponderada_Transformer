# Ponderada — Translation with a Transformer (PT→EN)

Repositório da **ponderada de programação** do Módulo 11 de Ciência da Computação: **“Atividade: Tradução usando Transformer com Controle de Versões”**.

Implementa um **Transformer encoder–decoder** seguindo o tutorial oficial do TensorFlow para **tradução automática de português → inglês**, baseado no paper _Attention Is All You Need_. O projeto foi executado e documentado em notebook, com análise crítica de resultados e comparação **CPU vs GPU**.



## Visão Geral do Projeto

- **Objetivo:** reproduzir e analisar um modelo Transformer para tradução PT→EN.
- **Base:** tutorial _Neural machine translation with a Transformer and Keras_ (TensorFlow).
- **Escopo:** treino do zero em ~50k pares PT–EN (TED Talks), avaliação simples e discussão técnica.
- **Entrega:** notebook `transformer.ipynb` + este `README.md`.


## Arquitetura Implementada

### Componentes
- **Encoder:** 4 camadas com Multi‑Head Attention + Feed‑Forward.
- **Decoder:** 4 camadas com self‑attention mascarada + cross‑attention.
- **Embedding:** dimensão **128** com **codificação posicional** sinusoidal.
- **Tokenização:** **subpalavras** via **BertTokenizer** (otimizado para PT/EN).
- **Atenção:** **8 cabeças** (multi‑head).

### Hiperparâmetros
```python
num_layers   = 4
d_model      = 128
dff          = 512
num_heads    = 8
dropout_rate = 0.1
batch_size   = 64
epochs       = 20
```

> Nota: parâmetros foram **reduzidos** em relação ao paper original (d_model=512, 6 camadas) para viabilizar a execução em recursos limitados.



## Dataset

- **Fonte:** TED Talks PT–EN
- **Treino:** ~50.000 pares de sentenças  
- **Validação:** ~1.100 exemplos  
- **Teste:** ~2.000 exemplos

O pipeline utiliza batching, prefetch e paralelização para melhor throughput. O tokenizador de subpalavras reduz OOV e melhora a generalização.



## Como Reproduzir

### 1) Requisitos
- Python 3.10+
- TensorFlow 2.x (GPU recomendado)
- CUDA/cuDNN compatíveis (se usar GPU)
- Demais libs listadas em `requirements.txt`

```bash
# ambiente (exemplo)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Executar o notebook
Abra e rode **`transformer.ipynb`** (Jupyter/VSCode/Colab).  
O notebook contém: preparação de dados, definição do modelo, treino, inferência e visualizações de atenção.

### 3) Exportar o modelo (opcional)
```python
# exemplo no notebook
tf.saved_model.save(model, "export/saved_model")
```



## Resultados de Treinamento (logs)

| Época | Loss | Acurácia |
|:----:|:----:|:--------:|
| 1 | 6.70 | 11.39% |
| 10 | 2.11 | 58.25% |
| 20 | 1.45 | 67.99% |

> Métricas adicionais como **BLEU/ROUGE/BERTScore** não foram computadas nesta versão (ver _Roadmap_).



## Exemplos de Tradução (PT → EN)

| Português | Predição | Ground Truth |
|---|---|---|
| este é um problema que temos que resolver. | this is a problem we have to solve . | this is a problem we have to solve . |
| os meus vizinhos ouviram sobre esta ideia. | my neighbors heard about this idea . | and my neighboring homes heard about this idea . |

**Observação:** em geral, o modelo apresenta boa fluência e cobertura; divergências ocorrem em detalhes semânticos e escolhas lexicais.



## Pontos Positivos

### Arquitetura & Desempenho
- **Paralelização superior:** processa tokens em paralelo (vs. RNNs sequenciais), acelerando o treino.
- **Dependências longas:** atenção captura relações de longo alcance sem degradação de gradiente.
- **Flexibilidade:** não assume estrutura temporal/espacial específica dos dados.
- **Generalização:** translitera termos raros (ex.: “triceratops” → “trigatotys”).

### Implementação
- **Código modular e limpo**, favorecendo reutilização.
- **Visualizações de atenção** para interpretabilidade.
- **Exportação como SavedModel** para servir em produção.
- **Caráter didático:** comentários e seções guiadas no notebook.



## Pontos Negativos

### Limitações Computacionais
- **Custo de memória:** atenção quadrática **O(n²)** limita sequências longas.
- **Hiperparâmetros reduzidos** (d_model=128) por restrição de hardware.
- **Dataset modesto** (~50k) para qualidade de produção.

### Limitações de Design
- **Sem transfer learning** (p.ex., mT5/T5/BART).
- **Sem beam search** (usa **greedy decoding**).
- **Positional encoding fixa** (sinusoidal) pode limitar sequências extensas.

### Limitações Práticas
- **Risco de overfitting:** regularização além de dropout não explorada.
- **Avaliação simplificada:** sem métricas padrão como **BLEU**.
- **Inferência sequencial no decoder**, limitando a velocidade.


## CPU vs GPU

### Treino em CPU
- **Tempo por época:** ~45–58s (com logs desta execução específica).
- **Gargalos:** multiplicações de matrizes e atenção em lote.
- **Viabilidade:** prototipagem/didática e datasets pequenos.

### Treino em GPU
- **Vantagens:** speedup **10–50×** em operações paralelizáveis.
- **Batch maior:** memória dedicada permite `batch_size > 64`.
- **Hiperparâmetros realistas:** viabiliza `d_model=512`, `num_layers=6`.
- **Requisito:** GPU **≥ 8 GB VRAM** recomendada.

**Recomendações:**  
- **CPU:** aprendizado/debugging.  
- **GPU:** treino sério.  
- **Cloud/TPUs:** Colab Pro/TPU para experimentação rápida.

## Estrutura do Repositório

```
├── README.md
├── transformer.ipynb      # notebook principal (execução e análise)
└── requirements.txt       # dependências do projeto
```

---

## 📜 Licença & Créditos

- **Baseado em:** Tutorial oficial do TensorFlow “Neural machine translation with a Transformer and Keras”.
- **Paper:** Vaswani et al., 2017 — _Attention Is All You Need_.

Este repositório tem propósito **educacional** e de **documentação**. Para casos de produção, recomenda‑se **fine‑tuning de modelos pré‑treinados** e pipelines de avaliação robustos.
