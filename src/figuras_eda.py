"""Figuras da EDA: tendências das features e espectros de envelope."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bearing import freq_defeito_ims
from features import espectro_envelope
from graficos import finalizar
from io_ims import FS, baixar_ims, ler_snapshot, listar_snapshots, sinal_rolamento

SAIDA = Path('results/figuras')
FREQS = freq_defeito_ims()


def fig_tendencias(df, ensaio):
    """RMS, curtose e amplitudes de envelope vs horas (1 linha por rolamento)."""
    painel = [('rms', 'RMS'), ('curtose', 'Curtose'), ('amp_bpfo', 'amp_bpfo (envelope)'),
              ('amp_bpfi', 'amp_bpfi (envelope)'), ('amp_bsf', 'amp_bsf = 2xBSF (envelope)')]
    d = df[df['ensaio'] == ensaio]
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for ax, (col, titulo) in zip(axs.ravel(), painel):
        for b, g in d.groupby('rolamento'):
            ax.plot(g['horas_desde_inicio'], g[col], lw=0.8, label=b)
        ax.set_title(titulo, fontsize=12, weight='bold')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
    axs[0, 0].legend(title='Rolamento')
    axs[2, 0].set_xlabel('Horas desde o início')
    axs[1, 1].set_xlabel('Horas desde o início')
    axs.ravel()[-1].axis('off')
    if ensaio == 'T3':   # marca o fim "oficial" do ensaio (4448 snapshots)
        h = d[d['idx'] == 4447]['horas_desde_inicio'].iloc[0]
        for ax in axs.ravel()[:5]:
            ax.axvline(h, color='k', ls='--', lw=0.8)
    plt.suptitle(f'EDA {ensaio}: tendências por rolamento', fontsize=14, weight='bold')
    plt.tight_layout()
    finalizar(SAIDA / f'eda_tendencias_{ensaio}.png', dpi=110)


def fig_envelope(raiz, ensaio, b, idx_ini, idx_fim):
    """Espectro de envelope no início e no fim da vida, com as frequências de defeito."""
    snaps = listar_snapshots(raiz, ensaio)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, idx, rot in zip(axs, [idx_ini, idx_fim], ['início da vida', 'fim da vida']):
        x = sinal_rolamento(ler_snapshot(snaps['arquivo'][idx], ensaio), ensaio, b)
        f, esp = espectro_envelope(x, FS)
        m = f <= 600
        ax.plot(f[m], esp[m], lw=0.8, color='tab:blue')
        for nome, f0, cor in [('BPFO', FREQS['BPFO'], 'tab:red'), ('BPFI', FREQS['BPFI'], 'tab:green'),
                              ('2xBSF', 2 * FREQS['BSF'], 'tab:orange')]:
            ax.axvline(f0, color=cor, ls='--', lw=1, label=f'{nome} = {f0:.0f} Hz')
        ax.set_title(f'{ensaio} B{b}: {rot} (idx {idx}, {snaps["horas_desde_inicio"][idx]:.0f} h)',
                     fontsize=12, weight='bold')
        ax.set_ylabel('Amplitude do envelope')
        ax.grid(alpha=0.3)
    axs[0].legend(loc='upper right')
    axs[1].set_xlabel('Frequência (Hz)')
    plt.tight_layout()
    finalizar(SAIDA / f'envelope_{ensaio}B{b}.png', dpi=110)


if __name__ == '__main__':
    SAIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet('data/features.parquet')
    df['rolamento'] = df['rolamento'].astype(str)
    for e in ['T1', 'T2', 'T3']:
        fig_tendencias(df, e)
    if len(sys.argv) > 1 and sys.argv[1] == 'tend':
        sys.exit()
    raiz = baixar_ims()
    n3 = int(sys.argv[1]) if len(sys.argv) > 1 else 6322
    fig_envelope(raiz, 'T1', 3, 100, 2155)
    fig_envelope(raiz, 'T1', 4, 100, 2155)
    fig_envelope(raiz, 'T2', 1, 50, 981)
    fig_envelope(raiz, 'T3', 3, 200, n3)
