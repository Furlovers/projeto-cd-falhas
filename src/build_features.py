"""Extrai as features de todos os snapshots dos 3 ensaios e grava data/features.parquet/.csv."""
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from bearing import freq_defeito_ims
from features import features_snapshot
from io_ims import FS, baixar_ims, falha_final, ler_snapshot, listar_snapshots, sinal_rolamento

FREQS = freq_defeito_ims()


def processa_arquivo(caminho, ensaio):
    """Lê um snapshot e devolve a lista de features dos 4 rolamentos."""
    x = ler_snapshot(caminho, ensaio)
    linhas = []
    for b in range(1, 5):
        feats = features_snapshot(sinal_rolamento(x, ensaio, b), FS, FREQS)
        feats['rolamento'] = b
        linhas.append(feats)
    return linhas


def main(n_jobs=-1):
    raiz = baixar_ims()
    t0 = time.time()
    tabelas = []
    for ensaio in ['T1', 'T2', 'T3']:
        snaps = listar_snapshots(raiz, ensaio)
        t1 = time.time()
        res = Parallel(n_jobs=n_jobs, batch_size=8)(
            delayed(processa_arquivo)(a, ensaio) for a in snaps['arquivo'])
        print(f'{ensaio}: {len(snaps)} arquivos em {time.time() - t1:.0f} s', flush=True)
        # Junta as features com os metadados do snapshot (res[i] -> snapshot i)
        linhas = []
        for (_, s), feats_lista in zip(snaps.iterrows(), res):
            for f in feats_lista:
                linhas.append({'ensaio': ensaio, 'rolamento': f.pop('rolamento'),
                               'timestamp': s['timestamp'], 'idx': s['idx'],
                               'horas_desde_inicio': s['horas_desde_inicio'],
                               'horas_ate_fim': s['horas_ate_fim'], **f})
        tabelas.append(pd.DataFrame(linhas))
    df = pd.concat(tabelas, ignore_index=True)
    df['rolamento'] = df['rolamento'].astype(int)
    df['snapshot_id'] = (df['ensaio'] + '_' + df['rolamento'].astype(str) + '_'
                         + df['timestamp'].dt.strftime('%Y.%m.%d.%H.%M.%S'))
    df['falha_final'] = [falha_final(e, b) for e, b in zip(df['ensaio'], df['rolamento'])]
    df['rolamento'] = 'B' + df['rolamento'].astype(str)   # B1..B4
    inicio = ['ensaio', 'rolamento', 'timestamp', 'idx', 'horas_desde_inicio',
              'horas_ate_fim', 'snapshot_id', 'falha_final']
    df = df[inicio + [c for c in df.columns if c not in inicio]]
    assert df['snapshot_id'].is_unique
    df.to_parquet('data/features.parquet', index=False)
    df.to_csv('data/features.csv', index=False)
    print(df.shape, f'tempo total: {time.time() - t0:.0f} s')
    return df


if __name__ == '__main__':
    main()
