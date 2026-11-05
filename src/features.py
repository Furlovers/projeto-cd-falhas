"""Features físicas de um snapshot de vibração: tempo, frequência (FFT) e envelope."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis, skew

# Bandas (Hz) para a energia espectral
BANDAS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000)]
# Banda do filtro passa-banda antes da demodulação por envelope
BANDA_ENV = (1000, 9000)
# Largura relativa da janela de busca em torno de cada frequência de defeito (+-3%)
TOL = 0.03


def features_tempo(x):
    """Estatísticas no domínio do tempo."""
    rms = np.sqrt(np.mean(x**2))
    pico = np.max(np.abs(x))
    media_abs = np.mean(np.abs(x))
    return {
        'rms': rms,
        'pico': pico,
        'pico_a_pico': np.max(x) - np.min(x),
        'curtose': kurtosis(x, fisher=False),   # curtose "normal" = 3
        'assimetria': skew(x),
        'fator_crista': pico / rms,
        'fator_forma': rms / media_abs,
        'fator_impulso': pico / media_abs,
        'desvio': np.std(x),
    }


def features_fft(x, fs):
    """Estatísticas do espectro de amplitude (FFT)."""
    n = len(x)
    espectro = np.abs(np.fft.rfft(x)) / n
    f = np.fft.rfftfreq(n, 1 / fs)
    esp_dc = espectro.copy()
    esp_dc[0] = 0                                  # ignora a componente DC
    saida = {
        'frequencia_pico': f[np.argmax(esp_dc)],
        'centroide_espectral': np.sum(f * esp_dc) / np.sum(esp_dc),
    }
    for lo, hi in BANDAS:
        mask = (f >= lo) & (f < hi)
        saida[f'energia_{lo // 1000}_{hi // 1000}k'] = np.sum(esp_dc[mask] ** 2)
    return saida


def espectro_envelope(x, fs, banda=BANDA_ENV):
    """Espectro do envelope: passa-banda -> |hilbert| -> remove média -> FFT."""
    sos = butter(4, banda, btype='bandpass', fs=fs, output='sos')
    xf = sosfiltfilt(sos, x)                       # filtro de fase zero
    env = np.abs(hilbert(xf))
    env = env - np.mean(env)
    esp = np.abs(np.fft.rfft(env)) / len(env)
    f = np.fft.rfftfreq(len(env), 1 / fs)
    return f, esp


def amp_na_faixa(f, esp, f0, tol=TOL):
    """Amplitude máxima do espectro numa faixa de +-tol em torno de f0."""
    mask = (f >= f0 * (1 - tol)) & (f <= f0 * (1 + tol))
    return esp[mask].max()


def features_envelope(x, fs, freqs):
    """Amplitudes do espectro do envelope nas frequências de defeito.

    freqs: dict com 'BPFO', 'BPFI', 'BSF' (Hz). Para o rolo usa-se 2xBSF.
    Cada amplitude é normalizada pela mediana do espectro do envelope.
    O sufixo _h12 soma o 1º e o 2º harmônico.
    """
    f, esp = espectro_envelope(x, fs)
    ref = np.median(esp[f > 0])                    # nível de ruído de fundo
    alvos = {'bpfo': freqs['BPFO'], 'bpfi': freqs['BPFI'], 'bsf': 2 * freqs['BSF']}
    saida = {}
    for nome, f0 in alvos.items():
        a1 = amp_na_faixa(f, esp, f0) / ref
        a2 = amp_na_faixa(f, esp, 2 * f0) / ref
        saida[f'amp_{nome}'] = a1
        saida[f'amp_{nome}_h12'] = a1 + a2
    return saida


def features_snapshot(sinal, fs, freqs):
    """Todas as features de um snapshot (vetor 1D de 20480 pontos)."""
    x = np.asarray(sinal, dtype=float)
    saida = {}
    saida.update(features_tempo(x))
    saida.update(features_fft(x, fs))
    saida.update(features_envelope(x, fs, freqs))
    return saida
