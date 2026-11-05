"""Obtenção dos dados do notebook por 3 rotas: cache local, download do cache ou reconstrução do bruto.

Arquivos de cache (pasta data/):
- features.parquet  -> 1 linha por snapshot x rolamento (21 features), gerado por build_features.py;
- cnn_cache.npz     -> sinal bruto (float16) dos snapshots usados pela CNN, gerado por cnn.construir_cache;
- dataset_rotulado.parquet -> sempre derivado de features.parquet com labels.rotular + split_temporal.
"""
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from io_ims import _procura_raiz, baixar_ims
from labels import rotular, split_temporal

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / 'data'
FEATURES = DATA / 'features.parquet'
CNN_CACHE = DATA / 'cnn_cache.npz'
ROTULADO = DATA / 'dataset_rotulado.parquet'
ARQUIVOS_CACHE = ['features.parquet', 'cnn_cache.npz']
NOME_ZIP = 'cache_ecm514.zip'


def cache_completo():
    """True se os 2 arquivos de cache existem em data/."""
    return FEATURES.exists() and CNN_CACHE.exists()


def bruto_disponivel():
    """Pasta raiz do dataset bruto, se já estiver em data/raw (não baixa nada); senão None."""
    return _procura_raiz(DATA / 'raw')


def url_valida(url):
    """A URL foi preenchida (não é o valor de exemplo com <...>)?"""
    return bool(url) and url.startswith('http') and '<' not in url


def baixar_cache(url):
    """Baixa o zip do cache (Google Drive via gdown, ou URL direta) e extrai em data/."""
    DATA.mkdir(parents=True, exist_ok=True)
    destino = DATA / NOME_ZIP
    if 'drive.google.com' in url:
        try:
            import gdown
        except ImportError:                          # o Colab já tem; fora dele, instala
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
            import gdown
        gdown.download(url, str(destino), quiet=False, fuzzy=True)
    else:
        urllib.request.urlretrieve(url, destino)
    with zipfile.ZipFile(destino) as z:
        nomes = [n for n in z.namelist() if Path(n).name in ARQUIVOS_CACHE]
        assert len(nomes) == len(ARQUIVOS_CACHE), f'zip sem os arquivos esperados: {z.namelist()}'
        for n in nomes:                              # extrai só os arquivos esperados, direto em data/
            with z.open(n) as src, open(DATA / Path(n).name, 'wb') as dst:
                dst.write(src.read())
    destino.unlink()


def criar_zip_cache():
    """Gera data/cache_ecm514.zip (para publicar no drive e usar como URL_CACHE)."""
    destino = DATA / NOME_ZIP
    with zipfile.ZipFile(destino, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for n in ARQUIVOS_CACHE:
            z.write(DATA / n, arcname=n)
    return destino


def gerar_rotulado():
    """Rótulos e split temporal (labels.py) aplicados a features.parquet -> dataset_rotulado.parquet."""
    df = split_temporal(rotular(pd.read_parquet(FEATURES)))
    df.to_parquet(ROTULADO)
    return df


def reconstruir_do_bruto(n_jobs=-1):
    """Rota 3: baixa o IMS (kagglehub), extrai as features e monta o cache da CNN."""
    import build_features
    import cnn
    tempos = {}
    cwd = os.getcwd()
    os.chdir(RAIZ)                         # build_features/cnn usam caminhos relativos (data/...)
    try:
        t0 = time.time()
        raiz = baixar_ims(DATA / 'raw')
        tempos['download_bruto_s'] = time.time() - t0
        print(f'dataset bruto em {raiz} ({tempos["download_bruto_s"]:.0f} s)')
        t0 = time.time()
        build_features.main(n_jobs=n_jobs)
        tempos['features_s'] = time.time() - t0
        gerar_rotulado()
        t0 = time.time()
        cnn.construir_cache(n_jobs=n_jobs)
        tempos['cache_cnn_s'] = time.time() - t0
    finally:
        os.chdir(cwd)
    return tempos


def obter_dados(usar_cache=True, url_cache=None):
    """Garante features.parquet, cnn_cache.npz e dataset_rotulado.parquet; devolve (rota, tempos)."""
    t0 = time.time()
    tempos = {}
    if usar_cache and cache_completo():
        rota = '1: cache local'
    elif usar_cache and url_valida(url_cache):
        try:
            baixar_cache(url_cache)
            rota = '2: cache baixado de URL_CACHE'
        except Exception as erro:  # noqa: BLE001 - se o download falhar, reconstrói do bruto
            print('download do cache falhou:', erro)
            rota = None
        if rota is None or not cache_completo():
            rota = '3: reconstruído do bruto (kagglehub)'
            tempos = reconstruir_do_bruto()
    else:
        rota = '3: reconstruído do bruto (kagglehub)'
        tempos = reconstruir_do_bruto()
    if not ROTULADO.exists() or ROTULADO.stat().st_mtime < FEATURES.stat().st_mtime:
        gerar_rotulado()
    tempos['total_s'] = time.time() - t0
    return rota, tempos
