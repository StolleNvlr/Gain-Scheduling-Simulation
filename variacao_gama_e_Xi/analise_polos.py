"""
Análise de polos para encontrar o valor EXATO de K_nom onde o sistema
começa a oscilar e onde perde a estabilidade.

Critério rigoroso:
  - Polos REAIS          → sem oscilação
  - Polos COMPLEXOS      → oscilação
  - |polo| < 1           → estável (sistema discreto)
  - |polo| >= 1          → instável
"""
import matplotlib
matplotlib.use('Agg')
import control as ct
import math as m
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros nominais
a_n = 1.6
b_n = 0.2
un = 1

s = ct.tf('s')


def calcular_planta_nominal(K_nom):
    """Calcula G(z) para velocidade nominal com K_nom dado."""
    eta = 1.0
    a = a_n * eta
    b = b_n * eta**2
    K = K_nom / eta**2

    Td = (2.4 * m.sqrt(0.2 * K) - 1.6) / (0.2 * K)

    G = (K * b * (1 + s * Td)) / (s**2 + s * (a + K * b * Td) + K * b)
    G_d = ct.c2d(G, 1)

    return G_d, Td


# =============================================
# VARREDURA FINA DE K_nom
# =============================================

K_sweep = np.concatenate([
    np.arange(0.01, 1, 0.01),
    np.arange(1, 50, 0.5),
    np.arange(50, 500, 2),
])

resultados = []

for K_nom in K_sweep:
    try:
        G_d, Td = calcular_planta_nominal(K_nom)
        polos = np.array(ct.poles(G_d))

        # Magnitude máxima dos polos (estabilidade)
        mag_max = max(np.abs(polos))

        # Parte imaginária máxima (oscilação)
        imag_max = max(np.abs(np.imag(polos)))

        # Classificação
        tem_oscilacao = imag_max > 1e-8  # polos complexos?
        estavel = mag_max < 1.0          # dentro do círculo unitário?

        resultados.append({
            'K_nom': K_nom,
            'Td': Td,
            'polos': polos,
            'mag_max': mag_max,
            'imag_max': imag_max,
            'oscila': tem_oscilacao,
            'estavel': estavel,
        })
    except Exception as e:
        resultados.append({
            'K_nom': K_nom,
            'Td': float('nan'),
            'polos': [],
            'mag_max': float('nan'),
            'imag_max': float('nan'),
            'oscila': None,
            'estavel': None,
        })


# =============================================
# ENCONTRAR FRONTEIRAS
# =============================================

# Fronteira de oscilação: primeiro K_nom onde polos ficam complexos
K_inicio_oscilacao = None
K_fim_oscilacao = None
for i in range(1, len(resultados)):
    r_prev = resultados[i-1]
    r_curr = resultados[i]
    if r_prev['oscila'] == False and r_curr['oscila'] == True and K_inicio_oscilacao is None:
        K_inicio_oscilacao = r_curr['K_nom']
    if r_prev['oscila'] == True and r_curr['oscila'] == False and K_fim_oscilacao is None:
        K_fim_oscilacao = r_curr['K_nom']

# Fronteira de instabilidade: primeiro K_nom onde |polo| >= 1
K_instabilidade = None
for r in resultados:
    if r['estavel'] == False:
        K_instabilidade = r['K_nom']
        break

print("=" * 65)
print("  ANALISE DE POLOS — FRONTEIRAS DO SISTEMA")
print("=" * 65)

if K_inicio_oscilacao:
    print(f"\n  >> O sistema COMECA A OSCILAR em K_nom ≈ {K_inicio_oscilacao:.3f}")
else:
    print(f"\n  >> O sistema NAO OSCILA em nenhum valor testado (0.01 a 498)")

if K_fim_oscilacao:
    print(f"  >> O sistema PARA DE OSCILAR em K_nom ≈ {K_fim_oscilacao:.3f}")

if K_instabilidade:
    print(f"  >> O sistema fica INSTAVEL a partir de K_nom ≈ {K_instabilidade:.3f}")
else:
    print(f"  >> O sistema permanece ESTAVEL em toda a faixa testada")

# Faixa segura (estável e sem oscilação)
estaveis_sem_osc = [r for r in resultados if r['estavel'] and not r['oscila']]
if estaveis_sem_osc:
    k_min_seguro = min(r['K_nom'] for r in estaveis_sem_osc)
    k_max_seguro = max(r['K_nom'] for r in estaveis_sem_osc)
    print(f"\n  FAIXA SEGURA (estavel + sem oscilacao):")
    print(f"     K_nom de {k_min_seguro:.3f} a {k_max_seguro:.3f}")

# Faixa com oscilação controlada (estável mas oscila)
estaveis_com_osc = [r for r in resultados if r['estavel'] and r['oscila']]
if estaveis_com_osc:
    k_min_osc = min(r['K_nom'] for r in estaveis_com_osc)
    k_max_osc = max(r['K_nom'] for r in estaveis_com_osc)
    print(f"\n  FAIXA COM OSCILACAO CONTROLADA (estavel + oscila):")
    print(f"     K_nom de {k_min_osc:.3f} a {k_max_osc:.3f}")

print(f"\n  Referencia: K original do MATLAB = 30")
print("=" * 65)


# =============================================
# GRÁFICOS
# =============================================

K_vals = [r['K_nom'] for r in resultados]
mag_vals = [r['mag_max'] for r in resultados]
imag_vals = [r['imag_max'] for r in resultados]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Análise de Polos vs K_nom (γ) — Detecção de Oscilação', fontsize=14, fontweight='bold')

# --- Subplot 1: Magnitude máxima dos polos ---
ax1 = axes[0]
colors1 = ['#2ca02c' if r['estavel'] else '#d62728' for r in resultados]
ax1.scatter(K_vals, mag_vals, c=colors1, s=8)
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='|polo| = 1 (fronteira de estabilidade)')
ax1.set_ylabel('|polo| máximo')
ax1.set_title('Magnitude máxima dos polos (estabilidade)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Parte imaginária dos polos ---
ax2 = axes[1]
colors2 = ['#ff7f0e' if r['oscila'] else '#2ca02c' for r in resultados]
ax2.scatter(K_vals, imag_vals, c=colors2, s=8)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('|Im(polo)| máximo')
ax2.set_title('Parte imaginária dos polos (Im > 0 = oscilação)')
ax2.grid(True, alpha=0.3)

# Marca a fronteira de oscilação
if K_inicio_oscilacao:
    ax2.axvline(x=K_inicio_oscilacao, color='red', linestyle=':', linewidth=2,
                label=f'Início oscilação: K_nom = {K_inicio_oscilacao:.2f}')
    ax2.legend(fontsize=10)

# --- Subplot 3: Mapa de classificação ---
ax3 = axes[2]
status_vals = []
colors3 = []
for r in resultados:
    if r['estavel'] and not r['oscila']:
        status_vals.append(2)
        colors3.append('#2ca02c')  # verde
    elif r['estavel'] and r['oscila']:
        status_vals.append(1)
        colors3.append('#ff7f0e')  # laranja
    elif not r['estavel']:
        status_vals.append(0)
        colors3.append('#d62728')  # vermelho
    else:
        status_vals.append(-1)
        colors3.append('#999999')

ax3.scatter(K_vals, status_vals, c=colors3, s=15)
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['Instável', 'Estável + Oscila', 'Estável + Sem Oscilação'])
ax3.set_xlabel('K_nom (γ)', fontsize=12)
ax3.set_title('Classificação do sistema')
ax3.grid(True, alpha=0.3)

if K_inicio_oscilacao:
    ax3.axvline(x=K_inicio_oscilacao, color='red', linestyle=':', linewidth=2)

plt.tight_layout()
plt.savefig('analise_polos.png', dpi=150, bbox_inches='tight')
print("\nGrafico salvo: analise_polos.png")


# =============================================
# MAPA DE POLOS NO PLANO Z (para valores selecionados)
# =============================================

K_exemplos = [0.01, 0.1, 1, 5, 10, 30, 50, 100, 200, 400]

fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle('Polos no Plano Z para diferentes K_nom', fontsize=14, fontweight='bold')

for idx, K_nom in enumerate(K_exemplos):
    ax = axes2[idx // 5, idx % 5]

    try:
        G_d, Td = calcular_planta_nominal(K_nom)
        polos = np.array(ct.poles(G_d))

        # Desenha o círculo unitário
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)

        # Plota os polos
        for p in polos:
            cor = '#d62728' if abs(p) >= 1 else ('#ff7f0e' if abs(np.imag(p)) > 1e-8 else '#2ca02c')
            ax.plot(np.real(p), np.imag(p), 'x', color=cor, markersize=12, markeredgewidth=2)

        oscila_str = "OSCILA" if max(abs(np.imag(polos))) > 1e-8 else "NAO OSCILA"
        estavel_str = "ESTAVEL" if max(abs(polos)) < 1 else "INSTAVEL"

        ax.set_title(f'K={K_nom}\n{estavel_str}, {oscila_str}', fontsize=8,
                     color='#2ca02c' if estavel_str == "ESTAVEL" else '#d62728')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)

    except Exception as e:
        ax.text(0.5, 0.5, f'ERRO', transform=ax.transAxes, ha='center')

plt.tight_layout()
plt.savefig('mapa_polos_plano_z.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: mapa_polos_plano_z.png")
