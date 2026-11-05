"""Funções de figura compartilhadas: salvar em results/figuras e mostrar no notebook."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def no_notebook():
    """True quando o backend do matplotlib é o inline do Jupyter/Colab."""
    return 'inline' in matplotlib.get_backend().lower()


def finalizar(arquivo=None, dpi=100, mostrar=True):
    """Salva a figura atual (se `arquivo`) e mostra no notebook; fora dele, só fecha."""
    if arquivo is not None:
        Path(arquivo).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(arquivo, dpi=dpi)
    if mostrar and no_notebook():
        plt.show()
    else:
        plt.close()


def matriz_confusao(cm, rotulos, titulo, ax=None):
    """Heatmap do seaborn com as contagens de uma matriz de confusão."""
    ax = ax or plt.gca()
    sns.heatmap(np.asarray(cm), annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=rotulos, yticklabels=rotulos, ax=ax)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title(titulo, fontsize=14, weight='bold')
    return ax


def espectro_envelope_painel(ax, f, esp, freqs, titulo, fmax=600):
    """Espectro do envelope até `fmax` Hz com as linhas de BPFO, BPFI e 2xBSF (e harmônicos 2x)."""
    m = f <= fmax
    ax.plot(f[m], esp[m], lw=0.8, color='tab:blue')
    for nome, f0, cor in [('BPFO', freqs['BPFO'], 'tab:red'), ('BPFI', freqs['BPFI'], 'tab:green'),
                          ('2xBSF', 2 * freqs['BSF'], 'tab:orange')]:
        ax.axvline(f0, color=cor, ls='--', lw=1, label=f'{nome} = {f0:.0f} Hz')
        if 2 * f0 <= fmax:
            ax.axvline(2 * f0, color=cor, ls=':', lw=0.8)
    ax.set_title(titulo, fontsize=12, weight='bold')
    ax.set_ylabel('Amplitude do envelope')
    ax.grid(alpha=0.3)
