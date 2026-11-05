"""Figuras da rotulagem: estágios (PCA + KMeans) ao longo da vida dos rolamentos que falharam."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from labels import FALHAS, estagios_kmeans, inicio_persistente, suavizar

SAIDA = Path('results/figuras')

df = pd.read_parquet('data/features.parquet')
df = df[df['rms'] > 0.01]                      # descarta snapshots com sinal colapsado

fig, axs = plt.subplots(len(FALHAS), 2, figsize=(14, 3.2 * len(FALHAS)))
for i, ((ensaio, rol), falha) in enumerate(FALHAS.items()):
    d = df[(df['ensaio'] == ensaio) & (df['rolamento'] == rol)].sort_values('idx').reset_index(drop=True)
    k, sil, rot, Z = estagios_kmeans(d)
    rot_s = suavizar(rot)
    ini = inicio_persistente(rot_s)
    h_ini = d.loc[ini, 'horas_desde_inicio'] if ini is not None else None
    h_fim = d['horas_desde_inicio'].iloc[-1]
    print(f'{ensaio}{rol} ({falha}): k={k} silhuetas={ {kk: round(v, 2) for kk, v in sil.items()} } '
          f'inicio idx={d.loc[ini, "idx"] if ini is not None else None} h={h_ini:.0f} ({h_fim - h_ini:.0f} h antes do fim)')
    ax = axs[i, 0]
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=d['horas_desde_inicio'], s=4, cmap='viridis')
    ax.set_title(f'{ensaio}-{rol} ({falha}): PCA colorido pelo tempo', fontsize=12, weight='bold')
    plt.colorbar(sc, ax=ax, label='horas')
    ax = axs[i, 1]
    ax.scatter(d['horas_desde_inicio'], rot, s=3, alpha=0.3, label='cluster')
    ax.plot(d['horas_desde_inicio'], rot_s, color='k', lw=1, label='moda móvel')
    if h_ini is not None:
        ax.axvline(h_ini, color='r', ls='--', label=f'início ({h_ini:.0f} h)')
    ax.set_title(f'{ensaio}-{rol}: cluster KMeans (k={k}) ao longo do tempo', fontsize=12, weight='bold')
    ax.set_xlabel('Horas desde o início')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(SAIDA / 'rotulagem_estagios.png', dpi=100)
