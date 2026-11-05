| Dataset | Técnica | Tipo de falha identificada | Taxa de detecção (TPR) | Falso positivo (FPR) | Acurácia por classe |
|---|---|---|---|---|---|
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Detecção D1: limiar μ+3σ (log RMS, curtose) | Normal × Falha (qualquer tipo) | 0,461 | 0,016 | Normal 0,984; Pista ext. 0,424; Pista int. 0,812; Rolo 0,333 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Detecção D2: PCA + KMeans (distância ao centróide) | Normal × Falha (qualquer tipo) | 0,935 | 0,013 | Normal 0,987; Pista ext. 0,945; Pista int. 0,970; Rolo 0,891 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Detecção D3: Random Forest binário (limiar de Youden) | Normal × Falha (qualquer tipo) | 0,994 | 0,053 | Normal 0,947; Pista ext. 0,990; Pista int. 1,000; Rolo 1,000 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Técnica 1: Random Forest (features) | Pista ext. / Pista int. / Rolo | 0,906 | 0,032 | Normal 0,968; Pista ext. 0,579; Pista int. 0,960; Rolo 0,739 |
| IMS T1+T2+T3, teste temporal (8404 N / 648 F) | Técnica 1: SVC (features) | Pista ext. / Pista int. / Rolo | 0,988 | 0,090 | Normal 0,910; Pista ext. 0,385; Pista int. 0,941; Rolo 1,000 |
| IMS, cache da CNN, teste (608 N / 648 F) | Técnica 2: CNN 1D (sinal bruto, 3 seeds) | Pista ext. / Pista int. / Rolo | 1,000 ± 0,000 | 0,170 ± 0,009 | Normal 0,830 ± 0,009; Pista ext. 0,848 ± 0,063; Pista int. 1,000 ± 0,000; Rolo 1,000 ± 0,000 |
| IMS, cache da CNN, teste (608 N / 648 F) | Comparação: MLPClassifier (features) | Pista ext. / Pista int. / Rolo | 1,000 | 0,151 | Normal 0,849; Pista ext. 0,628; Pista int. 1,000; Rolo 1,000 |
