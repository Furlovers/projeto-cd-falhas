"""Técnica 2: CNN 1D (Keras) sobre janelas do sinal bruto + comparação com MLPClassifier sobre as features.

Fluxo (as funções são chamadas pelo notebook; `python src/cnn.py` roda o experimento completo):
1. `construir_cache`  -> data/cnn_cache.npz (todas as falhas + Normal subamostrado, float16)
2. `preparar_dados`   -> janelas de 2048 pontos, split treino/validação/teste por snapshot
3. `criar_modelo` / `treinar_cnn` -> CNN pequena no estilo WDCNN (1ª convolução larga)
4. `avaliar`          -> métricas por janela e por snapshot (voto majoritário das 10 janelas)
5. `treinar_mlp`      -> Pipeline(StandardScaler, MLPClassifier) nos MESMOS snapshots do cache
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent))
from graficos import finalizar
from io_ims import baixar_ims, ler_snapshot, listar_snapshots, sinal_rolamento
from labels import CLASSES

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / 'data' / 'dataset_rotulado.parquet'
CACHE = RAIZ / 'data' / 'cnn_cache.npz'
FIGURAS = RAIZ / 'results' / 'figuras'

PASSO_NORMAL = 14          # 1 a cada 14 snapshots Normal (28036 / 14 ~ 2000)
TAM_JANELA = 2048
N_JANELAS = 10             # 10 x 2048 = 20480 pontos = 1 snapshot, sem sobreposição
SEEDS = [42, 43, 44]
NOMES = ['Normal', 'Pista ext.', 'Pista int.', 'Rolo']
CLASSE_ID = {c: i for i, c in enumerate(CLASSES)}


# ============================================================
# 1. Cache bruto
# ============================================================
def selecionar_snapshots(d, passo=PASSO_NORMAL):
    """Todas as falhas + Normal com passo fixo no tempo em cada (ensaio, rolamento, conjunto).

    O passo é aplicado separadamente em treino e teste, então a proporção 70/30 do split
    temporal é mantida. Como os rolamentos de um ensaio têm os mesmos idx Normal, os 4
    rolamentos caem nos mesmos arquivos (menos leitura de disco).
    """
    d = d.sort_values(['ensaio', 'rolamento', 'idx'])
    normal = d[d['classe'] == 'normal']
    pos = normal.groupby(['ensaio', 'rolamento', 'conjunto']).cumcount()
    sel = pd.concat([d[d['classe'] != 'normal'], normal[pos % passo == 0]])
    return sel.sort_values(['ensaio', 'rolamento', 'idx']).reset_index(drop=True)


def tabela_arquivos(raiz):
    """(ensaio, idx) -> caminho do arquivo do snapshot, para os 3 ensaios."""
    return pd.concat([listar_snapshots(raiz, e)[['ensaio', 'idx', 'arquivo']] for e in ['T1', 'T2', 'T3']])


def _ler_canais(arquivo, ensaio, rolamentos):
    """Lê um arquivo uma vez e devolve o canal x de cada rolamento pedido (float32)."""
    x = ler_snapshot(arquivo, ensaio)
    return np.stack([sinal_rolamento(x, ensaio, int(r[1:])) for r in rolamentos]).astype(np.float32)


def ler_sinais(sel, raiz=None, n_jobs=-1):
    """Lê o sinal bruto (N x 20480) dos snapshots de `sel`, na mesma ordem, em paralelo por arquivo."""
    raiz = raiz or baixar_ims()
    sel = sel.reset_index(drop=True).merge(tabela_arquivos(raiz), on=['ensaio', 'idx'], how='left')
    assert sel['arquivo'].notna().all(), 'snapshot sem arquivo correspondente'
    grupos = list(sel.groupby('arquivo', sort=False))
    res = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_ler_canais)(arq, g['ensaio'].iloc[0], g['rolamento'].tolist()) for arq, g in grupos)
    sinais = np.empty((len(sel), 20480), dtype=np.float32)
    for (_, g), s in zip(grupos, res):
        sinais[g.index.values] = s
    return sinais


def construir_cache(caminho=CACHE, passo=PASSO_NORMAL, n_jobs=-1):
    """Monta e grava o cache da CNN (float16). Devolve o tamanho em MB."""
    d = pd.read_parquet(DATASET)
    sel = selecionar_snapshots(d, passo)
    t0 = time.time()
    sinais = ler_sinais(sel, n_jobs=n_jobs)
    print(f'{len(sel)} snapshots lidos em {time.time() - t0:.0f} s')
    texto = {c: np.array(sel[c].tolist(), dtype='U') for c in ['snapshot_id', 'classe', 'conjunto', 'ensaio', 'rolamento']}
    np.savez(caminho, sinais=sinais.astype(np.float16), idx=sel['idx'].to_numpy(dtype=np.int64), **texto)
    mb = Path(caminho).stat().st_size / 2**20
    print(f'cache gravado em {caminho}: {mb:.0f} MB')
    return mb


def carregar_cache(caminho=CACHE):
    """Lê o cache -> (sinais float16 N x 20480, DataFrame de metadados na mesma ordem)."""
    z = np.load(caminho)
    meta = pd.DataFrame({c: z[c] for c in ['snapshot_id', 'classe', 'conjunto', 'ensaio', 'rolamento', 'idx']})
    return z['sinais'], meta


# ============================================================
# 2. Janelas, validação temporal e normalização
# ============================================================
def janelas(sinais):
    """(N, 20480) -> (N*10, 2048): cada snapshot vira 10 janelas consecutivas, sem sobreposição."""
    return np.asarray(sinais, dtype=np.float32).reshape(-1, TAM_JANELA)


def separar_validacao(meta, frac=0.2):
    """Marca como validação os últimos 20% (no tempo) do treino de cada (ensaio, rolamento, classe)."""
    meta = meta.copy()
    tr = meta[meta['conjunto'] == 'treino'].sort_values(['ensaio', 'rolamento', 'idx'])
    pos = tr.groupby(['ensaio', 'rolamento', 'classe']).cumcount()
    tam = tr.groupby(['ensaio', 'rolamento', 'classe'])['idx'].transform('size')
    meta['parte'] = meta['conjunto']
    meta.loc[tr.index[(pos >= (1 - frac) * tam).values], 'parte'] = 'validacao'
    return meta


def checar_vazamento(meta):
    """Nenhum snapshot pode aparecer em mais de uma parte (treino / validação / teste)."""
    grupos = [set(meta.loc[meta['parte'] == p, 'snapshot_id']) for p in ['treino', 'validacao', 'teste']]
    assert not grupos[0] & grupos[1], 'vazamento: snapshot em treino e validação'
    assert not grupos[0] & grupos[2], 'vazamento: snapshot em treino e teste'
    assert not grupos[1] & grupos[2], 'vazamento: snapshot em validação e teste'


def stats_globais(X_treino):
    """Média e desvio globais (um escalar cada), calculados só nas janelas de treino."""
    return float(X_treino.mean()), float(X_treino.std())


def normalizar(X, metodo, stats=None):
    """'global': (x - mu_treino) / sd_treino; 'janela': z-score de cada janela com a própria média/desvio."""
    if metodo == 'global':
        mu, sd = stats
        return ((X - mu) / sd).astype(np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def preparar_dados(sinais, meta, metodo, partes_treino=('treino',)):
    """Monta X (janelas normalizadas, com canal) e y para cada parte; janelas herdam a parte do snapshot."""
    meta = separar_validacao(meta)
    checar_vazamento(meta)
    dados = {}
    for p in ['treino', 'validacao', 'teste']:
        m = meta[meta['parte'] == p]
        X = janelas(sinais[m.index.values])
        y = np.repeat(m['classe'].map(CLASSE_ID).values, N_JANELAS)
        dados[p] = {'X': X, 'y': y, 'meta': m.reset_index(drop=True)}
    stats = stats_globais(dados['treino']['X']) if metodo == 'global' else None
    for p in dados:
        dados[p]['X'] = normalizar(dados[p]['X'], metodo, stats)[..., None]
    dados['stats'] = stats
    return dados


# ============================================================
# 3. Modelo e treino
# ============================================================
def criar_modelo(tam=TAM_JANELA, n_classes=len(CLASSES)):
    """CNN 1D pequena (estilo WDCNN): 1ª convolução larga + 2 blocos kernel 3 + GAP + softmax."""
    import keras
    from keras import layers
    entrada = keras.Input(shape=(tam, 1))
    x = entrada
    for filtros, kernel, stride in [(16, 64, 8), (32, 3, 1), (64, 3, 1)]:
        x = layers.Conv1D(filtros, kernel, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    saida = layers.Dense(n_classes, activation='softmax')(x)
    modelo = keras.Model(entrada, saida, name='cnn1d_wdcnn')
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo


def fixar_seed(seed):
    """Seeds de numpy, random e tensorflow + operações determinísticas."""
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def pesos_classes(y):
    """class_weight balanceado (n / (k * n_c)) no formato dict do Keras."""
    classes = np.unique(y)
    return dict(zip(classes.tolist(), compute_class_weight('balanced', classes=classes, y=y)))


def treinar_cnn(dados, seed, epocas=30, batch=64, paciencia=5, verbose=0):
    """Treina uma CNN com EarlyStopping na validação temporal. Devolve (modelo, histórico, segundos)."""
    import keras
    fixar_seed(seed)
    modelo = criar_modelo()
    parada = keras.callbacks.EarlyStopping(monitor='val_loss', patience=paciencia, restore_best_weights=True)
    t0 = time.time()
    hist = modelo.fit(dados['treino']['X'], dados['treino']['y'], epochs=epocas, batch_size=batch,
                      validation_data=(dados['validacao']['X'], dados['validacao']['y']),
                      class_weight=pesos_classes(dados['treino']['y']), callbacks=[parada],
                      shuffle=True, verbose=verbose)
    return modelo, hist.history, time.time() - t0


# ============================================================
# 4. Avaliação
# ============================================================
def voto_majoritario(prob):
    """Probabilidades por janela (N*10, 4) -> classe por snapshot. Empate: maior probabilidade média."""
    prob = prob.reshape(-1, N_JANELAS, prob.shape[-1])
    votos = np.zeros(prob.shape[::2])
    np.add.at(votos, (np.arange(len(prob))[:, None], prob.argmax(-1)), 1)
    return (votos + 1e-3 * prob.mean(1)).argmax(1)


def metricas(y, yhat):
    """Acurácia, F1 macro, acurácia (recall) por classe e TPR/FPR colapsando Normal x Falha."""
    cm = confusion_matrix(y, yhat, labels=range(len(CLASSES)))
    falha, pred_falha = y != 0, yhat != 0
    return {
        'acuracia': accuracy_score(y, yhat),
        'f1_macro': f1_score(y, yhat, average='macro', labels=range(len(CLASSES))),
        'acuracia_por_classe': {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None
                                for i, c in enumerate(CLASSES)},
        'tpr': float(pred_falha[falha].mean()),
        'fpr': float(pred_falha[~falha].mean()),
        'matriz_confusao': cm.tolist(),
    }


def avaliar(modelo, parte):
    """Métricas por janela e por snapshot (voto das 10 janelas) de uma parte (dict X, y, meta)."""
    prob = modelo.predict(parte['X'], batch_size=512, verbose=0)
    y_snap = parte['meta']['classe'].map(CLASSE_ID).values
    yhat_snap = voto_majoritario(prob)
    return {'janela': metricas(parte['y'], prob.argmax(1)), 'snapshot': metricas(y_snap, yhat_snap),
            'yhat_snapshot': yhat_snap}


def erros_por_grupo(meta, yhat):
    """Previsões por (ensaio, rolamento, classe real): mostra se o erro se concentra em algum rolamento."""
    d = meta[['ensaio', 'rolamento', 'classe']].copy()
    d['previsto'] = [CLASSES[i] for i in yhat]
    saida = {}
    for (e, r, c), g in d.groupby(['ensaio', 'rolamento', 'classe']):
        saida[f'{e}_{r}_{c}'] = {'n': int(len(g)), 'taxa_erro': float((g['previsto'] != c).mean()),
                                 'previsoes': g['previsto'].value_counts().to_dict()}
    return saida


def fpr_normal_ampliado(modelos, metodo, stats, cache_ids, n_jobs=-1, bloco=2000):
    """FPR por snapshot no Normal de teste completo (fora do cache), sem treinar de novo.

    Lê do disco os snapshots Normal de teste que não estão no cache, em blocos, e aplica cada modelo.
    """
    d = pd.read_parquet(DATASET)
    alvo = d[(d['classe'] == 'normal') & (d['conjunto'] == 'teste') & ~d['snapshot_id'].isin(cache_ids)]
    alvo = alvo.sort_values(['ensaio', 'rolamento', 'idx']).reset_index(drop=True)
    raiz = baixar_ims()
    alarmes = np.zeros((len(modelos), len(alvo)), dtype=bool)
    for ini in range(0, len(alvo), bloco):
        parte = alvo.iloc[ini:ini + bloco]
        X = normalizar(janelas(ler_sinais(parte, raiz, n_jobs)), metodo, stats)[..., None]
        for k, m in enumerate(modelos):
            alarmes[k, ini:ini + len(parte)] = voto_majoritario(m.predict(X, batch_size=1024, verbose=0)) != 0
    grupos = alvo['ensaio'] + '_' + alvo['rolamento']
    por_rolamento = {g: alarmes[:, (grupos == g).values].mean(1).tolist() for g in sorted(grupos.unique())}
    return {'n_snapshots': len(alvo), 'fpr_por_seed': alarmes.mean(1).tolist(),
            'fpr_media': media_desvio(alarmes.mean(1)), 'fpr_por_rolamento': por_rolamento}


def fpr_normal_completo_multi(configs, cache_ids, raiz, n_jobs=-1, bloco=2000):
    """Como `fpr_normal_ampliado`, mas para modelos com normalizações diferentes, lendo o disco uma vez só.

    configs: dict nome -> (modelo, metodo, stats). Devolve dict nome -> {'fpr', 'fpr_por_rolamento'}.
    """
    d = pd.read_parquet(DATASET)
    alvo = d[(d['classe'] == 'normal') & (d['conjunto'] == 'teste') & ~d['snapshot_id'].isin(cache_ids)]
    alvo = alvo.sort_values(['ensaio', 'rolamento', 'idx']).reset_index(drop=True)
    alarmes = {nome: np.zeros(len(alvo), dtype=bool) for nome in configs}
    for ini in range(0, len(alvo), bloco):
        parte = alvo.iloc[ini:ini + bloco]
        Xj = janelas(ler_sinais(parte, raiz, n_jobs))
        for nome, (m, metodo, stats) in configs.items():
            X = normalizar(Xj, metodo, stats)[..., None]
            alarmes[nome][ini:ini + len(parte)] = voto_majoritario(m.predict(X, batch_size=1024, verbose=0)) != 0
    grupos = (alvo['ensaio'] + '_' + alvo['rolamento']).values
    return {nome: {'n_snapshots': len(alvo), 'fpr': float(a.mean()),
                   'fpr_por_rolamento': {g: float(a[grupos == g].mean()) for g in sorted(set(grupos))}}
            for nome, a in alarmes.items()}


def media_desvio(valores):
    """'média ± desvio' de uma lista (desvio amostral entre seeds)."""
    v = np.asarray(valores, dtype=float)
    return {'media': float(v.mean()), 'desvio': float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def resumir_seeds(resultados):
    """Agrega as métricas de várias seeds (média ± desvio) por janela e por snapshot."""
    resumo = {}
    for nivel in ['janela', 'snapshot']:
        r = [res[nivel] for res in resultados]
        resumo[nivel] = {k: media_desvio([x[k] for x in r]) for k in ['acuracia', 'f1_macro', 'tpr', 'fpr']}
        resumo[nivel]['acuracia_por_classe'] = {
            c: media_desvio([x['acuracia_por_classe'][c] for x in r]) for c in CLASSES}
    return resumo


# ============================================================
# 5. MLPClassifier sobre as features (rede densa como referência para a CNN)
# ============================================================
COLS_MLP = ['rms', 'pico', 'pico_a_pico', 'curtose', 'assimetria', 'fator_crista', 'fator_forma',
            'fator_impulso', 'desvio', 'frequencia_pico', 'centroide_espectral',
            'energia_0_1k', 'energia_1_3k', 'energia_3_6k', 'energia_6_10k',
            'amp_bpfo', 'amp_bpfo_h12', 'amp_bpfi', 'amp_bpfi_h12', 'amp_bsf', 'amp_bsf_h12']
COLS_MLP_LOG = ['rms', 'pico', 'pico_a_pico', 'desvio', 'energia_0_1k', 'energia_1_3k', 'energia_3_6k',
                'energia_6_10k', 'amp_bpfo', 'amp_bpfo_h12', 'amp_bpfi', 'amp_bpfi_h12', 'amp_bsf', 'amp_bsf_h12']


def matriz_features(d):
    """Features físicas com log10 nas amplitudes e energias."""
    X = d[COLS_MLP].copy()
    X[COLS_MLP_LOG] = np.log10(X[COLS_MLP_LOG])
    return X


def treinar_mlp(ids_treino, ids_teste, random_state=42):
    """GridSearchCV pequeno de Pipeline(StandardScaler, MLPClassifier) nos snapshots do cache da CNN."""
    d = pd.read_parquet(DATASET)
    # Entradas e Saídas (treino ordenado no tempo -> folds contíguos com StratifiedKFold sem shuffle)
    tr = d[d['snapshot_id'].isin(ids_treino)].sort_values(['ensaio', 'rolamento', 'idx'])
    te = d[d['snapshot_id'].isin(ids_teste)].sort_values(['ensaio', 'rolamento', 'idx'])
    assert not set(tr['snapshot_id']) & set(te['snapshot_id']), 'vazamento: snapshot em treino e teste'
    X_tr, y_tr = matriz_features(tr), tr['classe'].map(CLASSE_ID).values
    X_te, y_te = matriz_features(te), te['classe'].map(CLASSE_ID).values
    # Definição
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp', MLPClassifier(max_iter=1000, random_state=random_state))])
    grade = {'mlp__hidden_layer_sizes': [(32,), (64,), (64, 32)], 'mlp__alpha': [1e-4, 1e-2, 1.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    busca = GridSearchCV(pipe, grade, cv=cv, scoring='f1_macro', n_jobs=-1)
    # Treinamento
    t0 = time.time()
    busca.fit(X_tr, y_tr)
    tempo = time.time() - t0
    # Avaliação
    yhat = busca.predict(X_te)
    print('MLP melhores parâmetros:', busca.best_params_, f'CV f1_macro={busca.best_score_:.3f}')
    print(classification_report(y_te, yhat, target_names=NOMES, digits=3))
    # FPR no Normal de teste completo (as features já existem para todos os snapshots)
    normal_te = d[(d['classe'] == 'normal') & (d['conjunto'] == 'teste')]
    fpr_total = float((busca.predict(matriz_features(normal_te)) != 0).mean())
    return {'melhores_parametros': {k: str(v) for k, v in busca.best_params_.items()},
            'cv_f1_macro': float(busca.best_score_), 'teste': metricas(y_te, yhat),
            'fpr_normal_teste_completo': fpr_total, 'n_normal_teste_completo': len(normal_te),
            'tempo_grid_s': tempo, 'yhat': yhat, 'y': y_te}


# ============================================================
# Figuras
# ============================================================
def fig_matrizes(cm_janela, cm_snapshot, titulo, arquivo):
    """Matrizes de confusão 4x4 (por janela e por snapshot) lado a lado."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cm, nome in zip(axs, [cm_janela, cm_snapshot], ['por janela', 'por snapshot (voto)']):
        sns.heatmap(np.asarray(cm), annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=NOMES, yticklabels=NOMES, ax=ax)
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title(f'{titulo} {nome}', fontsize=14, weight='bold')
    plt.tight_layout()
    finalizar(arquivo, dpi=120)


def fig_matriz_unica(cm, titulo, arquivo):
    """Uma matriz de confusão 4x4."""
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(np.asarray(cm), annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=NOMES, yticklabels=NOMES)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(titulo, fontsize=14, weight='bold')
    plt.tight_layout()
    finalizar(arquivo, dpi=120)


def fig_curvas(historicos, arquivo):
    """Curvas de loss e acurácia (treino e validação) de cada seed."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for seed, h in historicos.items():
        for ax, m in zip(axs, ['loss', 'accuracy']):
            linha, = ax.plot(h[m], label=f'treino (seed {seed})')
            ax.plot(h[f'val_{m}'], '--', color=linha.get_color(), label=f'validação (seed {seed})')
    for ax, nome in zip(axs, ['Loss (entropia cruzada)', 'Acurácia por janela']):
        ax.set_xlabel('época')
        ax.set_title(nome, fontsize=14, weight='bold')
        ax.legend(fontsize=8)
    plt.tight_layout()
    finalizar(arquivo, dpi=120)


def prob_falha_no_tempo(modelo, metodo, stats, ensaio, rolamento, n_jobs=-1):
    """Fração de janelas previstas como falha em todos os snapshots Normal de um rolamento (diagnóstico de FP)."""
    d = pd.read_parquet(DATASET)
    alvo = d[(d['ensaio'] == ensaio) & (d['rolamento'] == rolamento) & (d['classe'] == 'normal')]
    alvo = alvo.sort_values('idx').reset_index(drop=True)
    X = normalizar(janelas(ler_sinais(alvo, n_jobs=n_jobs)), metodo, stats)[..., None]
    pred = modelo.predict(X, batch_size=1024, verbose=0).argmax(1).reshape(-1, N_JANELAS)
    alvo['frac_janelas_falha'] = (pred != 0).mean(1)
    return alvo[['idx', 'conjunto', 'frac_janelas_falha']]


def fig_fp_no_tempo(curvas, arquivo):
    """Fração de janelas 'falha' ao longo da fase Normal dos rolamentos que falham (treino x teste)."""
    fig, axs = plt.subplots(1, len(curvas), figsize=(6.5 * len(curvas), 4))
    for ax, (nome, c) in zip(np.atleast_1d(axs), curvas.items()):
        suave = c['frac_janelas_falha'].rolling(15, center=True, min_periods=1).mean()
        ax.plot(c['idx'], suave, color='k', lw=1)
        for conj, cor in [('treino', 'tab:blue'), ('teste', 'tab:red')]:
            m = c['conjunto'] == conj
            ax.axvspan(c.loc[m, 'idx'].min(), c.loc[m, 'idx'].max(), color=cor, alpha=0.12, label=conj)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('idx do snapshot')
        ax.set_ylabel('fração de janelas previstas como falha')
        ax.set_title(f'{nome}: fase Normal', fontsize=14, weight='bold')
        ax.legend()
    plt.tight_layout()
    finalizar(arquivo, dpi=120)


# ============================================================
# Experimento completo
# ============================================================
def escolher_normalizacao(sinais, meta, seed=SEEDS[0]):
    """Treina 1 seed com cada normalização e escolhe a de maior F1 macro por janela na VALIDAÇÃO."""
    res = {}
    for metodo in ['global', 'janela']:
        dados = preparar_dados(sinais, meta, metodo)
        modelo, hist, seg = treinar_cnn(dados, seed)
        r = avaliar(modelo, dados['validacao'])
        res[metodo] = {'val_f1_macro_janela': r['janela']['f1_macro'], 'val_acuracia_janela': r['janela']['acuracia'],
                       'val_f1_macro_snapshot': r['snapshot']['f1_macro'], 'val_loss_min': float(min(hist['val_loss'])),
                       'epocas': len(hist['loss']), 'tempo_s': seg}
        print(metodo, {k: round(v, 4) for k, v in res[metodo].items()}, flush=True)
    escolha = max(res, key=lambda k: (res[k]['val_f1_macro_janela'], -res[k]['val_loss_min']))
    return escolha, res


def experimento_generalizacao(sinais, meta, metodo, seeds=SEEDS):
    """Treina sem nenhum snapshot do T3 (pista externa só do T2-B1) e testa em todos os snapshots T3 do cache."""
    meta_sem = meta.copy()
    t3 = meta_sem['ensaio'] == 'T3'
    meta_sem.loc[t3, 'conjunto'] = 'teste'                     # T3 inteiro vira teste
    meta_sem = meta_sem[t3 | (meta_sem['conjunto'] == 'treino')]  # fora do T3, só o treino original
    dados = preparar_dados(sinais, meta_sem, metodo)
    teste = dados['teste']
    assert (teste['meta']['ensaio'] == 'T3').all()
    saida = []
    for seed in seeds:
        modelo, _, _ = treinar_cnn(dados, seed)
        r = avaliar(modelo, teste)
        saida.append({'seed': seed, 'snapshot': r['snapshot'], 'janela': r['janela']})
        print('generalização seed', seed, 'acc por classe (snapshot):',
              {k: round(v, 3) for k, v in r['snapshot']['acuracia_por_classe'].items() if v is not None}, flush=True)
    return {'descricao': 'treino sem T3; teste = todos os snapshots T3 do cache (normal + pista externa)',
            'n_teste_snapshots': int(len(teste['meta'])),
            'por_seed': [{'seed': s['seed'], 'snapshot': s['snapshot']} for s in saida],
            'resumo_snapshot': {
                'recall_pista_externa': media_desvio([s['snapshot']['acuracia_por_classe']['pista_externa'] for s in saida]),
                'fpr': media_desvio([s['snapshot']['fpr'] for s in saida]),
                'tpr': media_desvio([s['snapshot']['tpr'] for s in saida])}}


def _sem_numpy(obj):
    """Remove arrays e converte tipos numpy para gravar em JSON."""
    if isinstance(obj, dict):
        return {k: _sem_numpy(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [_sem_numpy(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(generalizacao=True):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    saida = {}
    t_total = time.time()
    # Cache
    if not CACHE.exists():
        construir_cache()
    saida['cache_mb'] = CACHE.stat().st_size / 2**20
    sinais, meta = carregar_cache()
    saida['cache_contagens'] = meta.groupby(['classe', 'conjunto']).size().unstack().to_dict()
    print(meta.groupby(['classe', 'conjunto']).size().unstack(), flush=True)

    # Normalização escolhida pela validação
    metodo, comp = escolher_normalizacao(sinais, meta)
    saida['normalizacao'] = {'escolhida': metodo, 'comparacao_validacao_seed42': comp}
    print('normalização escolhida:', metodo, flush=True)

    # 3 seeds com a normalização escolhida
    dados = preparar_dados(sinais, meta, metodo)
    saida['n_janelas'] = {p: int(len(dados[p]['y'])) for p in ['treino', 'validacao', 'teste']}
    modelo = criar_modelo()
    modelo.summary()
    saida['n_parametros'] = int(modelo.count_params())
    modelos, historicos, resultados, por_seed = [], {}, [], []
    for seed in SEEDS:
        modelo, hist, seg = treinar_cnn(dados, seed)
        r = avaliar(modelo, dados['teste'])
        modelos.append(modelo)
        historicos[seed] = hist
        resultados.append(r)
        por_seed.append({'seed': seed, 'tempo_treino_s': seg, 'epocas': len(hist['loss']),
                         'melhor_epoca': int(np.argmin(hist['val_loss'])) + 1,
                         'janela': r['janela'], 'snapshot': r['snapshot'],
                         'erros_por_grupo_snapshot': erros_por_grupo(dados['teste']['meta'], r['yhat_snapshot'])})
        print(f'seed {seed}: {seg:.0f} s, {len(hist["loss"])} épocas | janela acc={r["janela"]["acuracia"]:.4f} '
              f'| snapshot acc={r["snapshot"]["acuracia"]:.4f} f1={r["snapshot"]["f1_macro"]:.4f}', flush=True)
        y_snap = dados['teste']['meta']['classe'].map(CLASSE_ID).values
        print(classification_report(y_snap, r['yhat_snapshot'], target_names=NOMES, digits=3))
    saida['por_seed'] = por_seed
    saida['resumo'] = resumir_seeds(resultados)
    saida['tempo_medio_por_seed_s'] = float(np.mean([p['tempo_treino_s'] for p in por_seed]))

    # Figuras (seed 42 como representativa)
    r0 = resultados[0]
    fig_matrizes(r0['janela']['matriz_confusao'], r0['snapshot']['matriz_confusao'],
                 'CNN 1D (teste)', FIGURAS / 'cnn_matriz_confusao.png')
    fig_curvas(historicos, FIGURAS / 'cnn_curvas_treino.png')

    # FPR no Normal de teste ampliado (sem retreinar)
    saida['fpr_normal_teste_ampliado'] = fpr_normal_ampliado(modelos, metodo, dados['stats'], set(meta['snapshot_id']))
    print('FPR ampliado:', saida['fpr_normal_teste_ampliado'], flush=True)

    # MLPClassifier nos mesmos snapshots
    mlp = treinar_mlp(meta.loc[meta['conjunto'] == 'treino', 'snapshot_id'],
                      meta.loc[meta['conjunto'] == 'teste', 'snapshot_id'])
    fig_matriz_unica(mlp['teste']['matriz_confusao'], 'MLP sobre features (teste)', FIGURAS / 'cnn_mlp_matriz_confusao.png')
    te = meta[meta['conjunto'] == 'teste'].sort_values(['ensaio', 'rolamento', 'idx'])
    mlp['erros_por_grupo'] = erros_por_grupo(te, mlp['yhat'])
    saida['mlp'] = mlp

    # Generalização (opcional)
    if generalizacao:
        saida['generalizacao_sem_T3'] = experimento_generalizacao(sinais, meta, metodo)

    saida['tempo_total_s'] = time.time() - t_total
    with open(RAIZ / 'results' / 'cnn.json', 'w', encoding='utf-8') as f:
        json.dump(_sem_numpy(saida), f, indent=2, ensure_ascii=False)
    print('tempo total:', round(saida['tempo_total_s']), 's')
    return saida


if __name__ == '__main__':
    main()
