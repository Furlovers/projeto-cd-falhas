"""Técnica 1: diagnóstico multiclasse com classificadores sobre features físicas.

Função principal: `rodar_classico()` (treina, avalia, gera figuras e grava results/classico.json).
As funções auxiliares de dados (`carregar_dataset`, `matriz_X`) também são usadas por `detect.py`.
"""
import json
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from graficos import finalizar
from labels import CLASSES

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / 'data' / 'dataset_rotulado.parquet'
RESULTADOS = RAIZ / 'results'
FIGURAS = RESULTADOS / 'figuras'
RANDOM_STATE = 42

# Todas as features numéricas de features.py (rms ... amp_bsf_h12)
COLS_FEATURES = ['rms', 'pico', 'pico_a_pico', 'curtose', 'assimetria', 'fator_crista', 'fator_forma',
                 'fator_impulso', 'desvio', 'frequencia_pico', 'centroide_espectral',
                 'energia_0_1k', 'energia_1_3k', 'energia_3_6k', 'energia_6_10k',
                 'amp_bpfo', 'amp_bpfo_h12', 'amp_bpfi', 'amp_bpfi_h12', 'amp_bsf', 'amp_bsf_h12']
# Amplitudes e energias (escala multiplicativa) entram em log10
COLS_LOG = ['rms', 'pico', 'pico_a_pico', 'desvio',
            'energia_0_1k', 'energia_1_3k', 'energia_3_6k', 'energia_6_10k',
            'amp_bpfo', 'amp_bpfo_h12', 'amp_bpfi', 'amp_bpfi_h12', 'amp_bsf', 'amp_bsf_h12']
# Features de nível absoluto (dependem do ganho/montagem de cada rolamento)
COLS_NIVEL = ['rms', 'pico', 'pico_a_pico', 'desvio',
              'energia_0_1k', 'energia_1_3k', 'energia_3_6k', 'energia_6_10k']
COLS_RELATIVAS = [c for c in COLS_FEATURES if c not in COLS_NIVEL]


# ----------------------------------------------------------------- dados
def matriz_X(d, colunas=COLS_FEATURES):
    """Seleciona as features e aplica log10 nas amplitudes e energias."""
    X = d[colunas].copy()
    cols_log = [c for c in colunas if c in COLS_LOG]
    X[cols_log] = np.log10(X[cols_log])
    return X


def carregar_dataset(caminho=DADOS):
    """Lê o dataset rotulado (rótulos e split temporal de labels.py) e devolve (treino, teste) ordenados no tempo."""
    df = pd.read_parquet(caminho)
    df = df.sort_values(['ensaio', 'rolamento', 'idx']).reset_index(drop=True)
    treino = df[df['conjunto'] == 'treino'].reset_index(drop=True)
    teste = df[df['conjunto'] == 'teste'].reset_index(drop=True)
    assert not set(treino['snapshot_id']) & set(teste['snapshot_id']), 'vazamento: snapshot em treino e teste'
    return treino, teste


def validacao_cruzada():
    """Validação cruzada temporal: StratifiedKFold sem shuffle sobre o treino ordenado (blocos contíguos por classe)."""
    return StratifiedKFold(n_splits=5, shuffle=False)


def salvar_json(obj, caminho):
    """Grava um dicionário em JSON convertendo tipos do numpy."""
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=conv)


# ----------------------------------------------------------------- modelos
def definir_modelos():
    """Listas base_estimators + param_grids para a comparação de modelos com GridSearchCV."""
    base_estimators = [
        ('KNN', KNeighborsClassifier()),
        ('Árvore', DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)),
        ('RandomForest', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1)),
        ('SVC', SVC(class_weight='balanced', random_state=RANDOM_STATE)),
    ]
    param_grids = [
        {'scaler': [StandardScaler(), 'passthrough'],
         'classifier__n_neighbors': [5, 15], 'classifier__weights': ['uniform', 'distance']},
        {'scaler': ['passthrough'],
         'classifier__max_depth': [5, 10, None], 'classifier__min_samples_leaf': [1, 10]},
        {'scaler': ['passthrough'],
         'classifier__n_estimators': [100], 'classifier__max_depth': [10, None],
         'classifier__min_samples_leaf': [1, 5]},
        {'scaler': [StandardScaler()],
         'classifier__C': [1, 10, 100], 'classifier__gamma': ['scale', 0.01]},
    ]
    return base_estimators, param_grids


def treinar_grid(X, y, cv):
    """GridSearchCV (f1_macro) para cada modelo; devolve dict nome -> resultado."""
    base_estimators, param_grids = definir_modelos()
    resultados = {}
    for (nome, est), grid in zip(base_estimators, param_grids):
        # Definição
        pipe = Pipeline([('scaler', StandardScaler()), ('classifier', est)])
        busca = GridSearchCV(pipe, grid, scoring='f1_macro', cv=cv, n_jobs=-1)
        # Treinamento
        t0 = time.time()
        busca.fit(X, y)
        tempo = time.time() - t0
        params = {k: (type(v).__name__ if not isinstance(v, (str, int, float, type(None))) else v)
                  for k, v in busca.best_params_.items()}
        print(f'{nome:13s} F1 macro CV = {busca.best_score_:.3f}  ({tempo:.0f} s)  {params}')
        resultados[nome] = {'busca': busca, 'modelo': busca.best_estimator_, 'f1_cv': busca.best_score_,
                            'params': params, 'tempo_s': tempo}
    return resultados


# ----------------------------------------------------------------- avaliação
def com_n_jobs(pipe, n):
    """Ajusta n_jobs do classificador (só o RF tem) sem mudar os hiperparâmetros do modelo."""
    if 'n_jobs' in pipe.named_steps['classifier'].get_params():
        pipe.set_params(classifier__n_jobs=n)
    return pipe


def binario_normal_falha(y_true, y_pred):
    """TPR e FPR colapsando as 3 falhas em uma classe (positivo = falha)."""
    yt = np.asarray(y_true) != 'normal'
    yp = np.asarray(y_pred) != 'normal'
    tpr = (yp & yt).sum() / yt.sum()
    fpr = (yp & ~yt).sum() / (~yt).sum()
    return float(tpr), float(fpr)


def figura_matriz_confusao(cm, titulo, arquivo, classes=CLASSES, mostrar=True):
    """Matriz de confusão em heatmap do seaborn (contagens)."""
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(titulo, fontsize=14, weight='bold')
    plt.tight_layout()
    finalizar(arquivo, dpi=100, mostrar=mostrar)


def metricas_multiclasse(y, y_pred, classes=CLASSES):
    """F1 macro, matriz 4x4, recall (acurácia) por classe e TPR/FPR Normal x Falha."""
    cm = confusion_matrix(y, y_pred, labels=classes)
    presentes = [c for c in classes if (np.asarray(y) == c).any()]
    recall = recall_score(y, y_pred, labels=presentes, average=None, zero_division=0)
    tpr, fpr = binario_normal_falha(y, y_pred)
    return {'f1_macro': float(f1_score(y, y_pred, labels=presentes, average='macro')),
            'acuracia_por_classe': dict(zip(presentes, recall.round(4).tolist())),
            'matriz_confusao': cm.tolist(), 'classes': classes,
            'TPR_falha': tpr, 'FPR_falha': fpr,
            'report': classification_report(y, y_pred, labels=presentes, output_dict=True, zero_division=0)}


def avaliar_multiclasse(modelo, X, y, nome, arquivo_fig=None, classes=CLASSES):
    """classification_report, matriz 4x4, recall por classe e TPR/FPR Normal x Falha."""
    # Avaliação
    y_pred = modelo.predict(X)
    print(f'\n===== {nome} =====')
    print(classification_report(y, y_pred, labels=classes, zero_division=0))
    saida = metricas_multiclasse(y, y_pred, classes)
    if arquivo_fig is not None:
        figura_matriz_confusao(np.array(saida['matriz_confusao']), f'Matriz de confusão: {nome}', arquivo_fig, classes)
    return saida


def erro_normal_por_rolamento(modelo, teste, X_te):
    """Fração dos snapshots Normais do teste previstos como falha, separada por ensaio-rolamento."""
    m = (teste['classe'] == 'normal').values
    errado = pd.Series(modelo.predict(X_te[m]) != 'normal')
    grupo = (teste.loc[m, 'ensaio'] + '-' + teste.loc[m, 'rolamento']).values
    return errado.groupby(grupo).mean().round(4).to_dict()


# ----------------------------------------------------------------- seleção de atributos
def importancias(X, y, modelo_rf):
    """Informação mútua (mutual_info_classif) e feature_importances_ do RF, com barplots ordenados."""
    mi = pd.Series(mutual_info_classif(X, y, random_state=RANDOM_STATE), index=X.columns).sort_values(ascending=False)
    imp = pd.Series(modelo_rf.named_steps['classifier'].feature_importances_, index=X.columns).sort_values(ascending=False)
    for serie, titulo, arquivo in [(mi, 'Informação mútua com a classe', 'classico_mutual_info.png'),
                                   (imp, 'Importância das features (Random Forest)', 'classico_importancia_rf.png')]:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=serie.values, y=serie.index, color='steelblue')
        plt.title(titulo, fontsize=14, weight='bold')
        plt.xlabel('valor')
        plt.ylabel('')
        plt.tight_layout()
        finalizar(FIGURAS / arquivo, dpi=100)
    return mi, imp


class ScoresFixos:
    """score_func do SelectKBest que devolve uma informação mútua já calculada (evita recalcular para cada k)."""

    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def __call__(self, X, y):
        assert X.shape[1] == len(self.scores)
        return self.scores


def _mi_no_bloco(X, y, idx, scaler):
    """Informação mútua no bloco de treino `idx`, depois do scaler do pipeline (como o SelectKBest a veria)."""
    Xb, yb = X.iloc[idx], y.iloc[idx]
    if not isinstance(scaler, str):
        Xb = clone(scaler).fit_transform(Xb)
    return mutual_info_classif(Xb, yb, random_state=RANDOM_STATE)


def _f1_fold(pipe, X, y, i_tr, i_va):
    """Ajusta o pipeline no fold de treino e devolve o F1 macro no fold de validação."""
    pipe.fit(X.iloc[i_tr], y.iloc[i_tr])
    return f1_score(y.iloc[i_va], pipe.predict(X.iloc[i_va]), average='macro')


def testar_selectkbest_rapido(modelo, X, y, X_te, y_te, cv, ks=(5, 10, 15)):
    """Mesmo experimento de `testar_selectkbest`, com a informação mútua calculada UMA vez por fold.

    A seleção continua dentro da CV (a MI de cada fold usa só o treino daquele fold), então os
    números coincidem com os de `testar_selectkbest`; a MI de cada fold é reutilizada para todos os k.
    """
    from joblib import Parallel, delayed
    folds = list(cv.split(X, y))
    blocos = [i_tr for i_tr, _ in folds] + [np.arange(len(X))]      # 5 folds + treino inteiro
    scaler = modelo.named_steps['scaler']
    mi = Parallel(n_jobs=-1)(delayed(_mi_no_bloco)(X, y, b, scaler) for b in blocos)

    def montar(k, scores):
        pipe = clone(modelo)
        if k != 'todas':
            pipe.steps.insert(1, ('selecao', SelectKBest(ScoresFixos(scores), k=k)))   # scaler -> seleção -> modelo
        return pipe

    rotulos = list(ks) + ['todas']
    combos = [(k, f) for k in rotulos for f in range(len(folds))]
    f1s = Parallel(n_jobs=-1)(delayed(_f1_fold)(com_n_jobs(montar(k, mi[f]), 1), X, y, *folds[f])
                              for k, f in combos)
    saida = {}
    for k in rotulos:
        f1_cv = float(np.mean([v for (kk, _), v in zip(combos, f1s) if kk == k]))
        pipe = com_n_jobs(montar(k, mi[-1]), -1).fit(X, y)
        f1_te = f1_score(y_te, pipe.predict(X_te), average='macro')
        saida[str(k)] = {'f1_cv': f1_cv, 'f1_teste': float(f1_te)}
        print(f'  k={k!s:5s} F1 CV={f1_cv:.3f}  F1 teste={f1_te:.3f}')
    return saida


def testar_selectkbest(modelo, X, y, X_te, y_te, cv, ks=(5, 10, 15)):
    """Insere SelectKBest(mutual_info, k) no pipeline e compara F1 macro (CV e teste) com todas as features."""
    mi = partial(mutual_info_classif, random_state=RANDOM_STATE)
    saida = {}
    for k in list(ks) + ['todas']:
        pipe = clone(modelo)
        if k != 'todas':
            pipe.steps.insert(1, ('selecao', SelectKBest(mi, k=k)))    # scaler -> seleção -> classificador
        # folds em paralelo (classificador sequencial); refit final com todos os núcleos
        f1_cv = cross_val_score(com_n_jobs(pipe, 1), X, y, scoring='f1_macro', cv=cv, n_jobs=-1).mean()
        com_n_jobs(pipe, -1).fit(X, y)
        f1_te = f1_score(y_te, pipe.predict(X_te), average='macro')
        saida[str(k)] = {'f1_cv': float(f1_cv), 'f1_teste': float(f1_te)}
        print(f'  k={k!s:5s} F1 CV={f1_cv:.3f}  F1 teste={f1_te:.3f}')
    return saida


# ----------------------------------------------------------------- generalização e identidade
def generalizacao_t3(df, modelos, colunas=COLS_FEATURES):
    """Treina com T1+T2 (pista externa só de T2-B1) e testa no T3 inteiro (normal + pista externa de T3-B3)."""
    tr = df[df['ensaio'] != 'T3'].sort_values(['ensaio', 'rolamento', 'idx'])
    te = df[df['ensaio'] == 'T3'].sort_values(['ensaio', 'rolamento', 'idx'])
    saida = {}
    for nome, modelo in modelos.items():
        m = com_n_jobs(clone(modelo), -1)
        m.fit(matriz_X(tr, colunas), tr['classe'])
        y_pred = m.predict(matriz_X(te, colunas))
        presentes = ['normal', 'pista_externa']
        rec = recall_score(te['classe'], y_pred, labels=presentes, average=None)
        tpr, fpr = binario_normal_falha(te['classe'], y_pred)
        cm = confusion_matrix(te['classe'], y_pred, labels=CLASSES)
        saida[nome] = {'acuracia_por_classe': dict(zip(presentes, rec.round(4).tolist())),
                       'f1_macro': float(f1_score(te['classe'], y_pred, labels=presentes, average='macro')),
                       'TPR_falha': tpr, 'FPR_falha': fpr, 'matriz_confusao': cm.tolist()}
        print(f'  {nome}: recall normal={rec[0]:.3f} pista_externa={rec[1]:.3f}  TPR={tpr:.3f} FPR={fpr:.3f}')
    return saida


def identidade_rolamento(df, colunas):
    """Acurácia (CV) de um RF que tenta adivinhar o rolamento (ensaio+rolamento) só com snapshots Normais.

    Se for alta, as features carregam a "assinatura" do rolamento/montagem, não só o estado de saúde.
    """
    d = df[df['classe'] == 'normal'].sort_values(['ensaio', 'rolamento', 'idx'])
    grupo = d['ensaio'] + '-' + d['rolamento']
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    acc = cross_val_score(rf, matriz_X(d, colunas), grupo, cv=validacao_cruzada(), scoring='accuracy').mean()
    return float(acc)


# ----------------------------------------------------------------- principal
def rodar_classico(salvar=True):
    """Executa toda a Técnica 1 e devolve o dicionário de resultados."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # Entradas e Saídas
    treino, teste = carregar_dataset()
    X_tr, y_tr = matriz_X(treino), treino['classe']
    X_te, y_te = matriz_X(teste), teste['classe']
    cv = validacao_cruzada()

    # Treinamento (GridSearch nos 4 modelos)
    print('--- GridSearchCV (f1_macro) ---')
    grid = treinar_grid(X_tr, y_tr, cv)
    tempos = {'gridsearch': time.time() - t_total}

    # Avaliação no teste
    avaliacao = {}
    for nome, r in grid.items():
        arq = FIGURAS / f'classico_matriz_{nome.lower().replace("á", "a")}.png'
        avaliacao[nome] = avaliar_multiclasse(r['modelo'], X_te, y_te, nome, arq)
    melhor = max(avaliacao, key=lambda n: grid[n]['f1_cv'])
    erros_normal = {n: erro_normal_por_rolamento(grid[n]['modelo'], teste, X_te) for n in grid}
    print('\nFração de Normais do teste classificados como falha, por rolamento:')
    print(pd.DataFrame(erros_normal).round(3))

    # Seleção de atributos
    print('\n--- Seleção de atributos ---')
    t0 = time.time()
    mi, imp = importancias(X_tr, y_tr, grid['RandomForest']['modelo'])
    selecao = {}
    for nome in ['RandomForest', 'SVC']:
        print(f' {nome}:')
        selecao[nome] = testar_selectkbest_rapido(grid[nome]['modelo'], X_tr, y_tr, X_te, y_te, cv)

    tempos['selecao_atributos'] = time.time() - t0

    # Generalização entre ensaios
    t0 = time.time()
    df_completo = pd.concat([treino, teste])
    modelos_destaque = {n: grid[n]['modelo'] for n in ['RandomForest', 'SVC']}
    print('\n--- Generalização: treino T1+T2, teste T3 (todas as features) ---')
    gen = generalizacao_t3(df_completo, modelos_destaque)
    print('--- Generalização: só features relativas (sem nível absoluto) ---')
    gen_rel = generalizacao_t3(df_completo, modelos_destaque, COLS_RELATIVAS)
    # referência: desempenho do split padrão restrito às linhas de T3
    ref_t3 = {}
    m_t3 = teste['ensaio'] == 'T3'
    for nome, modelo in modelos_destaque.items():
        y_pred = modelo.predict(X_te[m_t3])
        rec = recall_score(y_te[m_t3], y_pred, labels=['normal', 'pista_externa'], average=None)
        ref_t3[nome] = dict(zip(['normal', 'pista_externa'], rec.round(4).tolist()))

    tempos['generalizacao_T3'] = time.time() - t0

    # Identidade do rolamento a partir das features
    t0 = time.time()
    print('\n--- Identificação do rolamento só com snapshots normais ---')
    ident = {'todas': identidade_rolamento(df_completo, COLS_FEATURES),
             'relativas': identidade_rolamento(df_completo, COLS_RELATIVAS)}
    print(f'  acurácia (12 rolamentos): todas={ident["todas"]:.3f} relativas={ident["relativas"]:.3f}')
    tempos['identificacao_rolamento'] = time.time() - t0

    resultados = {
        'features': COLS_FEATURES, 'features_log10': COLS_LOG,
        'gridsearch': {n: {'f1_macro_cv': r['f1_cv'], 'melhores_params': r['params'], 'tempo_s': r['tempo_s']}
                       for n, r in grid.items()},
        'teste': avaliacao, 'melhor_modelo_cv': melhor, 'normal_previsto_falha_por_rolamento': erros_normal,
        'mutual_info': mi.round(4).to_dict(), 'importancia_rf': imp.round(4).to_dict(),
        'selectkbest_mutual_info': selecao,
        'generalizacao_T3': {'todas_features': gen, 'features_relativas': gen_rel,
                             'referencia_split_padrao_linhas_T3': ref_t3},
        'identificacao_rolamento_acuracia_cv': ident,
        'tempos_s': tempos, 'tempo_total_s': time.time() - t_total,
    }
    if salvar:
        salvar_json(resultados, RESULTADOS / 'classico.json')
    print(f'\nTempo total: {resultados["tempo_total_s"]:.0f} s')
    return resultados


if __name__ == '__main__':
    plt.switch_backend('Agg')
    rodar_classico()
