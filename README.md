# Detecção e Diagnóstico de Falhas em Rolamentos — IMS/NASA

Projeto da disciplina ECM514 – Ciência de Dados (IMT). A partir de sinais reais de vibração do **IMS/NASA Bearing Dataset**, o trabalho:

1. **detecta anomalias** (Normal × Falha), com TPR, FPR, curva ROC e antecedência da detecção;
2. **diagnostica o tipo de falha** (Normal, Pista externa, Pista interna, Rolo), com matriz de confusão e acurácia por classe.

## Sumário das entregas

| Entrega | Onde |
|---|---|
| Notebook com as análises (executável no Colab) | [`ECM514_Projeto2.ipynb`](ECM514_Projeto2.ipynb) · [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Furlovers/projeto-cd-falhas/blob/main/ECM514_Projeto2.ipynb) |
| Dados brutos | [IMS Bearing Dataset (Kaggle)](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset), baixado automaticamente pelo notebook |
| Dados processados (features + sinais usados pela CNN) | [`cache_ecm514.zip`](https://github.com/Furlovers/projeto-cd-falhas/releases/download/v1.0-dados/cache_ecm514.zip) (release `v1.0-dados`, 108 MB) |
| Relatório técnico | [`Relatorio_Tecnico.docx`](Relatorio_Tecnico.docx) |
| Resultados (métricas e figuras) | [`results/`](results/) |
| Link Vídeo Youtube | https://youtu.be/nEFErnTXIk8 |

## Técnicas

- **Detecção:** D1, limiar μ+3σ em RMS e curtose; D2, PCA + KMeans ajustados só em dados normais (distância ao centróide); D3, classificadores binários com limiar escolhido na curva ROC.
- **Técnica 1:** Random Forest e SVC (com KNN e Árvore de Decisão na comparação) sobre features físicas do sinal: estatísticas no tempo, energia espectral e amplitude do espectro do envelope nas frequências de defeito (BPFO, BPFI, 2×BSF).
- **Técnica 2:** CNN 1D (Keras) sobre janelas do sinal bruto, comparada a um MLP sobre as features.
- **Validação:** split temporal (70% inicial de cada rolamento e classe no treino), validação cruzada em blocos contíguos e ajuste de normalização e PCA só no treino, para evitar vazamento entre snapshots vizinhos.

## Tabela-resumo dos resultados (conjunto de teste)

| Dataset | Técnica | Tipo de falha identificada | Taxa de detecção (TPR) | Falso positivo (FPR) | Acurácia por classe |
|---|---|---|---|---|---|
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Detecção D1: limiar μ+3σ (log RMS, curtose) | Normal × Falha (qualquer tipo) | 0,461 | 0,016 | Normal 0,984; Pista ext. 0,424; Pista int. 0,812; Rolo 0,333 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Detecção D2: PCA + KMeans (distância ao centróide) | Normal × Falha (qualquer tipo) | 0,935 | 0,013 | Normal 0,987; Pista ext. 0,945; Pista int. 0,970; Rolo 0,891 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Detecção D3: Random Forest binário (limiar de Youden) | Normal × Falha (qualquer tipo) | 0,994 | 0,053 | Normal 0,947; Pista ext. 0,990; Pista int. 1,000; Rolo 1,000 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Técnica 1: Random Forest (features) | Pista ext. / Pista int. / Rolo | 0,906 | 0,032 | Normal 0,968; Pista ext. 0,579; Pista int. 0,960; Rolo 0,739 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Técnica 1: SVC (features) | Pista ext. / Pista int. / Rolo | 0,988 | 0,090 | Normal 0,910; Pista ext. 0,385; Pista int. 0,941; Rolo 1,000 |
| IMS, cache da CNN, teste (608 N / 648 F) | Técnica 2: CNN 1D (sinal bruto, 3 seeds) | Pista ext. / Pista int. / Rolo | 1,000 ± 0,000 | 0,170 ± 0,009 | Normal 0,830 ± 0,009; Pista ext. 0,848 ± 0,063; Pista int. 1,000 ± 0,000; Rolo 1,000 ± 0,000 |
| IMS, cache da CNN, teste (608 N / 648 F) | Comparação: MLPClassifier (features) | Pista ext. / Pista int. / Rolo | 1,000 | 0,151 | Normal 0,849; Pista ext. 0,628; Pista int. 1,000; Rolo 1,000 |

Matrizes de confusão completas: [`results/figuras/`](results/figuras/) (`classico_matriz_randomforest.png`, `classico_matriz_svc.png`, `cnn_matriz_confusao.png`).

## Como reproduzir

**No Google Colab (recomendado):** abra o notebook pelo botão acima e use *Ambiente de execução → Executar tudo*. A primeira célula clona este repositório, instala o que faltar (inclusive `scikit-learn==1.6.1`) e baixa o cache de dados da release. A execução completa leva cerca de 15–25 min em CPU.

**Localmente (Python 3.12):**

```bash
git clone https://github.com/Furlovers/projeto-cd-falhas.git
cd projeto-cd-falhas
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute ECM514_Projeto2.ipynb --ExecutePreprocessor.timeout=3600
```

**Rotas de dados** (célula *0. Setup*, variável `USAR_CACHE`):
1. cache local em `data/`, se existir;
2. download do cache publicado na release (padrão);
3. com `USAR_CACHE = False`: download do dataset bruto do Kaggle (~1,6 GB, sem credencial) e reconstrução das features e do cache da CNN.

As seeds são fixas (`random_state=42`). Execuções repetidas no mesmo ambiente dão os mesmos números; em outro hardware ou versão do TensorFlow, os resultados da CNN podem variar levemente.

## Estrutura

```
ECM514_Projeto2.ipynb   notebook com as análises e saídas
ECM514_Projeto2.py      mesmo notebook em formato texto (jupytext)
Relatorio_Tecnico.docx  relatório técnico do trabalho
src/                    funções de apoio: leitura (io_ims), frequências de defeito (bearing), features,
                        rótulos (labels), detecção (detect), classificadores (classic), CNN (cnn), cache e gráficos
results/                métricas (JSON), tabela-resumo e figuras
requirements.txt        dependências
```
