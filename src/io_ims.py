"""Download, leitura e listagem dos snapshots do dataset IMS/NASA Bearing."""
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_ID = 'vinayak123tyagi/bearing-dataset'
URL_NASA = 'https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip'
FS = 20480  # Hz

# Pasta de cada ensaio (relativa à raiz do dataset), nº de canais e coluna de cada rolamento.
# No T1 há 2 canais (x, y) por rolamento: usamos só o x (colunas 0, 2, 4, 6).
ENSAIOS = {
    'T1': {'pasta': '1st_test', 'n_canais': 8, 'colunas': [0, 2, 4, 6]},
    'T2': {'pasta': '2nd_test', 'n_canais': 4, 'colunas': [0, 1, 2, 3]},
    'T3': {'pasta': '3rd_test', 'n_canais': 4, 'colunas': [0, 1, 2, 3]},
}

# Falha ao final de cada ensaio (rolamentos que não aparecem aqui não falharam)
FALHAS = {
    ('T1', 3): 'pista_interna',
    ('T1', 4): 'rolo',
    ('T2', 1): 'pista_externa',
    ('T3', 3): 'pista_externa',
}

_PADRAO_NOME = re.compile(r'^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$')


def falha_final(ensaio, rolamento):
    """Tipo de falha do rolamento (1..4) ao final do ensaio, ou 'nenhuma'."""
    return FALHAS.get((ensaio, rolamento), 'nenhuma')


def parse_timestamp(nome):
    """Converte o nome do arquivo (ex.: 2003.10.22.12.06.24) em datetime."""
    return datetime.strptime(Path(nome).name[:19], '%Y.%m.%d.%H.%M.%S')


def _raiz_valida(raiz):
    """True se a pasta contém as 3 pastas de ensaio."""
    return raiz.is_dir() and all((raiz / e['pasta']).is_dir() for e in ENSAIOS.values())


def _procura_raiz(base):
    """Procura (recursivamente) a pasta que contém 1st_test, 2nd_test e 3rd_test."""
    base = Path(base)
    if not base.exists():
        return None
    for p in [base] + [d for d in base.rglob('*') if d.is_dir() and d.name == '1st_test']:
        cand = p if p.name != '1st_test' else p.parent
        if _raiz_valida(cand):
            return cand
    return None


def baixar_ims(destino='data/raw'):
    """Baixa o dataset (se ainda não existir) e devolve a pasta raiz.

    Rota 1: kagglehub (funciona sem token, dataset público).
    Rota 2 (reserva): espelho NASA/PHM (zip -> IMS.7z/rar, precisa de py7zr/rarfile).
    É idempotente: se os dados já estão em `destino`, não baixa de novo.
    """
    destino = Path(destino)
    raiz = _procura_raiz(destino)
    if raiz is not None:
        return raiz

    destino.mkdir(parents=True, exist_ok=True)
    try:
        os.environ['KAGGLEHUB_CACHE'] = str((destino / 'kagglehub').resolve())
        import kagglehub
        raiz = _procura_raiz(kagglehub.dataset_download(KAGGLE_ID))
    except Exception as erro:  # noqa: BLE001 - qualquer falha cai para a rota reserva
        print('kagglehub falhou:', erro)
        raiz = _baixar_nasa(destino)
    if raiz is None:
        raise RuntimeError('Não foi possível obter o dataset IMS.')
    return raiz


def _baixar_nasa(destino):
    """Rota reserva: espelho NASA/PHM (~1,1 GB). Os arquivos aninhados são tratados como dados."""
    import requests
    pasta = destino / 'nasa'
    pasta.mkdir(exist_ok=True)
    arq_zip = pasta / '4.Bearings.zip'
    if not arq_zip.exists():
        with requests.get(URL_NASA, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(arq_zip, 'wb') as f:
                for bloco in r.iter_content(1 << 20):
                    f.write(bloco)
    with zipfile.ZipFile(arq_zip) as z:
        z.extractall(pasta)
    for arq in pasta.rglob('*'):           # IMS.7z ou IMS.rar dentro do zip
        if arq.suffix == '.7z':
            import py7zr
            with py7zr.SevenZipFile(arq) as z:
                z.extractall(arq.parent)
        elif arq.suffix == '.rar':
            import rarfile
            with rarfile.RarFile(arq) as z:
                z.extractall(arq.parent)
    return _procura_raiz(pasta)


def pasta_ensaio(raiz, ensaio):
    """Pasta que realmente contém os snapshots do ensaio.

    Procura a subpasta com mais arquivos de nome-timestamp: o T1/T2 vêm em
    'Xst_test/Xst_test' e o T3 em '3rd_test/4th_test/txt'.
    """
    base = Path(raiz) / ENSAIOS[ensaio]['pasta']
    melhor, n_melhor = None, 0
    for d in [base] + [p for p in base.rglob('*') if p.is_dir()]:
        n = sum(1 for f in d.iterdir() if f.is_file() and _PADRAO_NOME.match(f.name))
        if n > n_melhor:
            melhor, n_melhor = d, n
    return melhor


def listar_snapshots(raiz, ensaio):
    """DataFrame com os snapshots do ensaio, em ordem temporal."""
    pasta = pasta_ensaio(raiz, ensaio)
    nomes = sorted(f.name for f in pasta.iterdir() if f.is_file() and _PADRAO_NOME.match(f.name))
    df = pd.DataFrame({'arquivo': [str(pasta / n) for n in nomes],
                       'timestamp': [parse_timestamp(n) for n in nomes]})
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.insert(0, 'ensaio', ensaio)
    df.insert(1, 'idx', np.arange(len(df)))
    t0, t1 = df['timestamp'].iloc[0], df['timestamp'].iloc[-1]
    df['horas_desde_inicio'] = (df['timestamp'] - t0).dt.total_seconds() / 3600
    df['horas_ate_fim'] = (t1 - df['timestamp']).dt.total_seconds() / 3600
    return df


def ler_snapshot(caminho, ensaio=None):
    """Lê um snapshot (texto separado por tab, 20480 linhas) -> array (20480, n_canais)."""
    x = pd.read_csv(caminho, sep='\t', header=None, dtype=np.float64).to_numpy()
    if ensaio is not None:
        assert x.shape[1] == ENSAIOS[ensaio]['n_canais'], f'{caminho}: {x.shape}'
    return x


def sinal_rolamento(x, ensaio, rolamento):
    """Extrai o canal (vetor 1D) do rolamento 1..4 de um snapshot lido."""
    return x[:, ENSAIOS[ensaio]['colunas'][rolamento - 1]]


if __name__ == '__main__':
    raiz = baixar_ims()
    print('raiz:', raiz)
    for e in ENSAIOS:
        df = listar_snapshots(raiz, e)
        print(e, len(df), df['timestamp'].iloc[0], df['timestamp'].iloc[-1],
              f"{df['horas_desde_inicio'].iloc[-1]:.1f} h")
