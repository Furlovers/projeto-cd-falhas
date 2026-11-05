"""Detecção de anomalias (Normal vs Falha): D1 limiar estatístico, D2 PCA + KMeans, D3 supervisionado binário.

Função principal: `rodar_deteccao()` (ajusta os detectores, avalia no teste, mede antecedência e
alarmes falsos na vida completa, gera figuras e grava results/deteccao.json).
D1 e D2 são ajustados **só com os snapshots Normais do treino**.
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve, silhouette_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from classic import (FIGURAS, RAIZ, RANDOM_STATE, RESULTADOS, carregar_dataset, matriz_X,
                     salvar_json, validacao_cruzada)
from graficos import finalizar
from labels import AMACIAMENTO, CORTE_NORMAL, FALHAS, INICIO, rotular

# aviso inofensivo do scipy/lbfgs na LogisticRegression ("Unknown solver options: iprint")
warnings.filterwarnings('ignore', message='.*iprint.*')

PERSISTENCIA = 10          # snapshots seguidos acima do limiar para contar como alarme persistente


def alvo_binario(d):
    """1 = falha (qualquer tipo), 0 = normal."""
    return (d['classe'] != 'normal').astype(int).values


# ----------------------------------------------------------------- D1: limiar estatístico
def ajustar_d1(X_normal):
    """Média e desvio de log10(RMS) e curtose no Normal do treino."""
    return {c: (X_normal[c].mean(), X_normal[c].std()) for c in ['rms', 'curtose']}


def score_d1(params, X):
    """Maior z-score entre log(RMS) e curtose; alarme quando passa de 3 (mu + 3 sigma)."""
    z = [(X[c] - mu) / sd for c, (mu, sd) in params.items()]
    return np.maximum(*z).values


# ----------------------------------------------------------------- D2: PCA + KMeans
def escolher_k(Z, ks=range(2, 13)):
    """Cotovelo (inércia) e silhueta para cada k; k escolhido = maior silhueta."""
    inercias, silhuetas = {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        inercias[k] = km.inertia_
        # silhueta numa amostra (o cálculo completo é O(n^2))
        silhuetas[k] = silhouette_score(Z, km.labels_, sample_size=5000, random_state=RANDOM_STATE)
    k = max(silhuetas, key=silhuetas.get)
    return k, inercias, silhuetas


def ajustar_d2(X_normal):
    """StandardScaler + PCA(90% da variância) + KMeans, tudo ajustado só no Normal do treino."""
    # Definição
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.9, random_state=RANDOM_STATE))])
    # Treinamento
    Z = pipe.fit_transform(X_normal)
    k, inercias, silhuetas = escolher_k(Z)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    modelo = {'pipe': pipe, 'kmeans': km, 'k': k, 'inercias': inercias, 'silhuetas': silhuetas,
              'n_componentes': int(pipe.named_steps['pca'].n_components_)}
    modelo['limiar'] = float(np.percentile(score_d2(modelo, X_normal), 99))
    return modelo


def score_d2(modelo, X):
    """Distância ao centróide mais próximo no espaço PCA."""
    return modelo['kmeans'].transform(modelo['pipe'].transform(X)).min(axis=1)


def figura_k_d2(modelo):
    """Cotovelo e silhueta do KMeans do D2."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ks = list(modelo['inercias'])
    axs[0].plot(ks, list(modelo['inercias'].values()), 'o-')
    axs[0].set_title('Cotovelo (inércia)', fontsize=14, weight='bold')
    axs[1].plot(ks, list(modelo['silhuetas'].values()), 'o-')
    axs[1].axvline(modelo['k'], color='r', ls='--', label=f'k = {modelo["k"]}')
    axs[1].set_title('Silhueta', fontsize=14, weight='bold')
    axs[1].legend()
    for ax in axs:
        ax.set_xlabel('k')
    plt.tight_layout()
    finalizar(FIGURAS / 'deteccao_d2_escolha_k.png', dpi=100)


# ----------------------------------------------------------------- D3: supervisionado binário
def modelos_d3():
    """LogisticRegression e RandomForest binários com class_weight='balanced'."""
    return {
        'D3_LogReg': Pipeline([('scaler', StandardScaler()),
                               ('classifier', LogisticRegression(class_weight='balanced', max_iter=2000,
                                                                 random_state=RANDOM_STATE))]),
        'D3_RF': Pipeline([('scaler', 'passthrough'),
                           ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                                 random_state=RANDOM_STATE, n_jobs=-1))]),
    }


def limiar_youden(y, prob):
    """Threshold que maximiza TPR - FPR (índice J de Youden)."""
    fpr, tpr, thr = roc_curve(y, prob)
    i = np.argmax(tpr - fpr)
    return float(thr[i]), float(tpr[i]), float(fpr[i])


def ajustar_d3(modelo, X, y, cv):
    """Escolhe o threshold no treino (cross_val_predict + Youden) e depois treina no treino inteiro."""
    prob_oof = cross_val_predict(modelo, X, y, cv=cv, method='predict_proba')[:, 1]
    thr, tpr_cv, fpr_cv = limiar_youden(y, prob_oof)
    modelo.fit(X, y)
    return {'modelo': modelo, 'limiar': thr, 'TPR_cv': tpr_cv, 'FPR_cv': fpr_cv}


# ----------------------------------------------------------------- métricas
def metricas_binarias(y, score, limiar):
    """TPR, FPR, matriz 2x2 e AUC (do score contínuo)."""
    alarme = (score > limiar).astype(int)
    cm = confusion_matrix(y, alarme, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_c, tpr_c, _ = roc_curve(y, score)
    return {'TPR': tp / (tp + fn), 'FPR': fp / (fp + tn), 'AUC': auc(fpr_c, tpr_c),
            'matriz_2x2 [[TN, FP], [FN, TP]]': cm.tolist(), 'limiar': float(limiar)}


def figura_roc(y, scores):
    """Curvas ROC no teste de todos os detectores no mesmo gráfico."""
    plt.figure(figsize=(6.5, 6))
    for nome, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=f'{nome} (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC no teste: Normal x Falha', fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    finalizar(FIGURAS / 'deteccao_roc.png', dpi=100)


def figura_matrizes_2x2(metricas):
    """Matrizes 2x2 de todos os detectores lado a lado."""
    fig, axs = plt.subplots(1, len(metricas), figsize=(4 * len(metricas), 3.6))
    for ax, (nome, m) in zip(axs, metricas.items()):
        cm = np.array(m['matriz_2x2 [[TN, FP], [FN, TP]]'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['normal', 'alarme'], yticklabels=['normal', 'falha'])
        ax.set_title(nome, fontsize=14, weight='bold')
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
    plt.tight_layout()
    finalizar(FIGURAS / 'deteccao_matrizes.png', dpi=100)


# ----------------------------------------------------------------- vida completa
def carregar_vida():
    """Vida completa dos 12 rolamentos (features.parquet) com a classe atribuída por rotular()."""
    df = rotular(pd.read_parquet(RAIZ / 'data' / 'features.parquet'))
    return df.sort_values(['ensaio', 'rolamento', 'idx']).reset_index(drop=True)


def primeiro_alarme_persistente(alarme, n=PERSISTENCIA):
    """Posição do início da primeira sequência de n alarmes seguidos (None se não houver)."""
    seguidos = pd.Series(alarme.astype(int)).rolling(n).sum().values
    pos = np.where(seguidos >= n)[0]
    return int(pos[0] - n + 1) if len(pos) else None


def antecedencia(vida, detectores):
    """Primeiro alarme persistente (horas antes do fim) de cada detector nos 4 rolamentos que falharam."""
    saida = {}
    for (ensaio, rol), falha in FALHAS.items():
        d = vida[(vida['ensaio'] == ensaio) & (vida['rolamento'] == rol) & (vida['idx'] >= AMACIAMENTO)]
        d = d.reset_index(drop=True)
        X = matriz_X(d)
        ini = d.index[d['idx'] >= INICIO[(ensaio, rol)]][0]
        linha = {'falha': falha, 'inicio_rotulado_idx': INICIO[(ensaio, rol)],
                 'inicio_rotulado_h_antes_fim': round(float(d.loc[ini, 'horas_ate_fim']), 1)}
        for nome, (fscore, limiar) in detectores.items():
            pos = primeiro_alarme_persistente(fscore(X) > limiar)
            if pos is None:
                linha[nome] = None
                continue
            linha[nome] = {'idx': int(d.loc[pos, 'idx']),
                           'h_antes_fim': round(float(d.loc[pos, 'horas_ate_fim']), 1),
                           'antes_do_corte_normal': bool(d.loc[pos, 'idx'] < CORTE_NORMAL[ensaio])}
        saida[f'{ensaio}-{rol}'] = linha
    return saida


def alarmes_falsos(vida, detectores, teste_ids):
    """Taxa de alarme nos rolamentos sadios: antes do corte (total e só teste) e depois do corte."""
    falhados = {f'{e}-{r}' for e, r in FALHAS}
    sadio = ~(vida['ensaio'] + '-' + vida['rolamento']).isin(falhados) & (vida['idx'] >= AMACIAMENTO)
    antes = sadio & (vida['idx'] < vida['ensaio'].map(CORTE_NORMAL))
    depois = sadio & (vida['idx'] >= vida['ensaio'].map(CORTE_NORMAL))
    so_teste = antes & vida['snapshot_id'].isin(teste_ids)
    X = matriz_X(vida)
    saida = {}
    for nome, (fscore, limiar) in detectores.items():
        alarme = fscore(X) > limiar
        por_ensaio = {e: float(alarme[antes & (vida['ensaio'] == e)].mean()) for e in CORTE_NORMAL}
        saida[nome] = {'antes_corte_total': float(alarme[antes].mean()),
                       'antes_corte_so_teste': float(alarme[so_teste].mean()),
                       'antes_corte_por_ensaio': por_ensaio,
                       'depois_corte': float(alarme[depois].mean())}
    return saida


def figura_score_d2(vida, modelo_d2, ant):
    """Score do D2 ao longo da vida de cada rolamento que falhou, com limiar, início rotulado e 1º alarme."""
    fig, axs = plt.subplots(len(FALHAS), 1, figsize=(11, 3 * len(FALHAS)))
    for ax, ((ensaio, rol), falha) in zip(axs, FALHAS.items()):
        d = vida[(vida['ensaio'] == ensaio) & (vida['rolamento'] == rol)]
        s = score_d2(modelo_d2, matriz_X(d))
        h = d['horas_desde_inicio'].values
        ax.semilogy(h, s, lw=0.7, color='steelblue', label='score D2')
        ax.axhline(modelo_d2['limiar'], color='k', ls=':', label='limiar (p99 Normal treino)')
        alarme = ant[f'{ensaio}-{rol}']['D2']
        if alarme is not None:
            h_al = d.loc[d['idx'] == alarme['idx'], 'horas_desde_inicio'].iloc[0]
            ax.axvline(h_al, color='orange', ls='-', lw=2, label='1º alarme persistente')
        h_ini = d.loc[d['idx'] >= INICIO[(ensaio, rol)], 'horas_desde_inicio'].iloc[0]
        ax.axvline(h_ini, color='r', ls='--', label='início rotulado')
        ax.set_title(f'{ensaio}-{rol} ({falha}): score D2 ao longo da vida', fontsize=14, weight='bold')
        ax.set_ylabel('distância ao centróide')
        ax.legend(fontsize=8, loc='upper left')
    axs[-1].set_xlabel('Horas desde o início do ensaio')
    plt.tight_layout()
    finalizar(FIGURAS / 'deteccao_score_d2_vida.png', dpi=100)


# ----------------------------------------------------------------- principal
def rodar_deteccao(salvar=True):
    """Executa D1, D2 e D3, avalia no teste, mede antecedência e alarmes falsos; devolve os resultados."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Entradas e Saídas
    treino, teste = carregar_dataset()
    X_tr, y_tr = matriz_X(treino), alvo_binario(treino)
    X_te, y_te = matriz_X(teste), alvo_binario(teste)
    X_normal = X_tr[y_tr == 0]
    cv = validacao_cruzada()

    # Treinamento: D1 e D2 só no Normal do treino
    p_d1 = ajustar_d1(X_normal)
    d2 = ajustar_d2(X_normal)
    figura_k_d2(d2)
    print(f'D2: {d2["n_componentes"]} componentes PCA, k = {d2["k"]}, limiar = {d2["limiar"]:.3f}')
    detectores = {'D1': (lambda X: score_d1(p_d1, X), 3.0),
                  'D2': (lambda X: score_d2(d2, X), d2['limiar'])}

    # Treinamento: D3 supervisionado (threshold escolhido no treino)
    d3 = {nome: ajustar_d3(m, X_tr, y_tr, cv) for nome, m in modelos_d3().items()}
    for nome, r in d3.items():
        detectores[nome] = (lambda X, m=r['modelo']: m.predict_proba(X)[:, 1], r['limiar'])
        print(f'{nome}: limiar Youden (CV treino) = {r["limiar"]:.3f}  TPR_cv={r["TPR_cv"]:.3f} FPR_cv={r["FPR_cv"]:.3f}')

    # Avaliação no teste
    scores_te = {nome: f(X_te) for nome, (f, _) in detectores.items()}
    metricas = {nome: metricas_binarias(y_te, scores_te[nome], lim) for nome, (_, lim) in detectores.items()}
    for nome, m in metricas.items():
        print(f'{nome:10s} TPR={m["TPR"]:.3f} FPR={m["FPR"]:.4f} AUC={m["AUC"]:.3f}')
    figura_roc(y_te, scores_te)
    figura_matrizes_2x2(metricas)

    # TPR por tipo de falha no teste (qual falha cada detector pega)
    tpr_tipo = {nome: {c: float((scores_te[nome][teste['classe'] == c] > lim).mean())
                       for c in ['pista_externa', 'pista_interna', 'rolo']}
                for nome, (_, lim) in detectores.items()}

    # Vida completa: antecedência (D1, D2) e alarmes falsos
    vida = carregar_vida()
    nao_sup = {k: detectores[k] for k in ['D1', 'D2']}
    ant = antecedencia(vida, nao_sup)
    falsos = alarmes_falsos(vida, nao_sup, set(teste['snapshot_id']))
    figura_score_d2(vida, d2, ant)
    print(pd.DataFrame(ant).T)
    print(pd.DataFrame(falsos).T)

    resultados = {
        'D1': {'mu_sigma': {c: [float(m), float(s)] for c, (m, s) in p_d1.items()},
               'obs': 'rms em log10; alarme se z(log RMS) > 3 ou z(curtose) > 3'},
        'D2': {'k': d2['k'], 'n_componentes_pca': d2['n_componentes'], 'limiar_p99': d2['limiar'],
               'silhuetas': d2['silhuetas'], 'inercias': d2['inercias']},
        'D3': {n: {'limiar_youden': r['limiar'], 'TPR_cv': r['TPR_cv'], 'FPR_cv': r['FPR_cv']} for n, r in d3.items()},
        'teste': metricas, 'TPR_por_tipo_teste': tpr_tipo,
        'antecedencia': ant, 'alarmes_falsos_sadios': falsos,
        'tempo_total_s': time.time() - t0,
    }
    if salvar:
        salvar_json(resultados, RESULTADOS / 'deteccao.json')
    print(f'Tempo total: {resultados["tempo_total_s"]:.0f} s')
    return resultados


if __name__ == '__main__':
    plt.switch_backend('Agg')
    rodar_deteccao()
