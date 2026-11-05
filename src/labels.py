"""Rotulagem por estágios de degradação: StandardScaler -> PCA -> KMeans (Jolliffe & Cadima 2016; MacQueen 1967)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Rolamentos que falharam ao final de cada ensaio (fonte: documentação do IMS)
FALHAS = {('T1', 'B3'): 'pista_interna', ('T1', 'B4'): 'rolo',
          ('T2', 'B1'): 'pista_externa', ('T3', 'B3'): 'pista_externa'}

# Features usadas no agrupamento (amplitudes e energias em escala log)
COLS_CLUSTER = ['rms', 'curtose', 'fator_crista', 'amp_bpfo', 'amp_bpfi', 'amp_bsf',
                'energia_0_1k', 'energia_1_3k', 'energia_3_6k', 'energia_6_10k']
COLS_LOG = ['rms', 'amp_bpfo', 'amp_bpfi', 'amp_bsf',
            'energia_0_1k', 'energia_1_3k', 'energia_3_6k', 'energia_6_10k']


def matriz_cluster(d):
    """Seleciona as colunas do agrupamento e aplica log nas de escala multiplicativa."""
    X = d[COLS_CLUSTER].copy()
    X[COLS_LOG] = np.log10(X[COLS_LOG])
    return X


def suavizar(rotulos, janela=15):
    """Moda móvel dos clusters ao longo do tempo (reduz a troca de cluster snapshot a snapshot)."""
    s = pd.Series(rotulos)
    return s.rolling(janela, center=True, min_periods=1).apply(lambda v: np.bincount(v.astype(int)).argmax()).astype(int).values


def estagios_kmeans(d, ks=range(2, 6), random_state=42):
    """Agrupa os snapshots de um rolamento e devolve (k escolhido, silhuetas, rótulos, coordenadas PCA).

    Os clusters são renumerados por ordem de aparição média no tempo: 0 = mais cedo (saudável).
    """
    X = matriz_cluster(d)
    silhuetas = {}
    modelos = {}
    for k in ks:
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)),
                         ('kmeans', KMeans(n_clusters=k, n_init=10, random_state=random_state))])
        rot = pipe.fit_predict(X)
        silhuetas[k] = silhouette_score(pipe[:-1].transform(X), rot)
        modelos[k] = (pipe, rot)
    k = max(silhuetas, key=silhuetas.get)
    pipe, rot = modelos[k]
    # renumera clusters pela posição média no tempo
    ordem = pd.Series(d['idx'].values).groupby(rot).mean().sort_values().index
    mapa = {c: i for i, c in enumerate(ordem)}
    rot = np.array([mapa[c] for c in rot])
    return k, silhuetas, rot, pipe[:-1].transform(X)


def inicio_persistente(rotulos, minimo=0.9):
    """Primeiro índice a partir do qual >= `minimo` dos snapshots seguintes estão fora do cluster saudável (0)."""
    fora = (np.asarray(rotulos) != 0).astype(float)
    frac_futuro = np.cumsum(fora[::-1])[::-1] / np.arange(len(fora), 0, -1)
    candidatos = np.where(frac_futuro >= minimo)[0]
    return int(candidatos[0]) if len(candidatos) else None


# ---- Rótulos adotados: início da falha por rolamento e fim da classe Normal por ensaio ----
# Início da falha: 2 de 3 indicadores (RMS, curtose, envelope) acima de mu+3sigma por 10 snapshots
INICIO = {('T1', 'B3'): 1819, ('T1', 'B4'): 1605, ('T2', 'B1'): 532, ('T3', 'B3'): 5497}
# Fim da classe Normal por ensaio (menor início individual no ensaio - 5% da vida)
CORTE_NORMAL = {'T1': 1375, 'T2': 483, 'T3': 5181}
AMACIAMENTO = 10        # primeiros snapshots descartados (run-in)
CLASSES = ['normal', 'pista_externa', 'pista_interna', 'rolo']


def rotular(df):
    """Adiciona a coluna `classe` (normal / tipo de falha / excluido) a partir dos inícios adotados em INICIO."""
    df = df[df['rms'] > 0.01].copy()                     # remove snapshots com sinal colapsado
    classe = pd.Series('excluido', index=df.index)
    normal = (df['idx'] >= AMACIAMENTO) & (df['idx'] < df['ensaio'].map(CORTE_NORMAL))
    classe[normal] = 'normal'
    for (ensaio, rol), inicio in INICIO.items():
        m = (df['ensaio'] == ensaio) & (df['rolamento'] == rol) & (df['idx'] >= inicio)
        classe[m] = FALHAS[(ensaio, rol)]
    df['classe'] = classe
    return df


def split_temporal(df, frac_treino=0.7):
    """Split temporal: em cada (ensaio, rolamento, classe), os primeiros 70% vão para treino e o resto para teste."""
    df = df[df['classe'] != 'excluido'].sort_values(['ensaio', 'rolamento', 'idx']).copy()
    pos = df.groupby(['ensaio', 'rolamento', 'classe']).cumcount()
    tam = df.groupby(['ensaio', 'rolamento', 'classe'])['idx'].transform('size')
    df['conjunto'] = np.where(pos < frac_treino * tam, 'treino', 'teste')
    treino, teste = df[df['conjunto'] == 'treino'], df[df['conjunto'] == 'teste']
    assert not set(treino['snapshot_id']) & set(teste['snapshot_id']), 'vazamento: snapshot em treino e teste'
    return df
