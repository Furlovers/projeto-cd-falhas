"""Frequências características de defeito de rolamentos (BPFO, BPFI, BSF, FTF)."""
import numpy as np

# Parâmetros do rolamento Rexnord ZA-2115 do dataset IMS
IMS_N = 16          # número de rolos
IMS_D_ROLO = 0.331  # diâmetro do rolo (in)
IMS_D_PRIM = 2.815  # diâmetro primitivo (in)
IMS_ANGULO = 15.17  # ângulo de contato (graus)
IMS_RPM = 2000      # rotação do eixo
IMS_FS = 20480      # taxa de amostragem (Hz)


def freq_defeito(fr, n, d, D, angulo):
    """Retorna as frequências de defeito (Hz) de um rolamento.

    fr: rotação do eixo em Hz; n: nº de rolos; d: diâmetro do rolo;
    D: diâmetro primitivo (mesma unidade de d); angulo: ângulo de contato em graus.
    """
    r = (d / D) * np.cos(np.deg2rad(angulo))
    bpfo = (n / 2) * fr * (1 - r)           # pista externa
    bpfi = (n / 2) * fr * (1 + r)           # pista interna
    bsf = (D / (2 * d)) * fr * (1 - r**2)   # rolo (spin)
    ftf = (fr / 2) * (1 - r)                # gaiola
    return {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf, 'FTF': ftf}


def freq_defeito_ims():
    """Frequências de defeito do IMS a 2000 rpm."""
    return freq_defeito(IMS_RPM / 60, IMS_N, IMS_D_ROLO, IMS_D_PRIM, IMS_ANGULO)


if __name__ == '__main__':
    for nome, valor in freq_defeito_ims().items():
        print(f'{nome}: {valor:.2f} Hz')
