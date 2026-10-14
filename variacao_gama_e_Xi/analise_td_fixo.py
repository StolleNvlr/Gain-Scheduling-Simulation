"""
Análise com Td FIXO para encontrar as fronteiras de oscilação e instabilidade.

Quando Td é fixo, o ganho K altera os polos livremente, permitindo
que se tornem complexos (oscilação) ou saiam do círculo unitário (instabilidade).
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

# Td fixo calculado para K=30 (valor original do MATLAB)
K_ref = 30
Td_fixo = (2.4 * m.sqrt(0.2 * K_ref) - 1.6) / (0.2 * K_ref)
print(f"Td fixo (calculado para K=30): Td = {Td_fixo:.6f}")


def calcular_planta_td_fixo(K_nom):
    """G(z) para velocidade nominal com K_nom dado e Td FIXO."""
    eta = 1.0
    a = a_n * eta       # 1.6
    b = b_n * eta**2    # 0.2
    K = K_nom / eta**2  # K_nom

    G = (K * b * (1 + s * Td_fixo)) / (s**2 + s * (a + K * b * Td_fixo) + K * b)
    G_d = ct.c2d(G, 1)
    return G_d


# =============================================
# VARREDURA FINA
# =============================================

K_sweep = np.concatenate([
    np.arange(0.01, 1, 0.01),
    np.arange(1, 50, 0.25),
    np.arange(50, 200, 1),
    np.arange(200, 1000, 5),
    np.arange(1000, 5000, 50),
])

resultados = []

for K_nom in K_sweep:
    try:
        G_d = calcular_planta_td_fixo(K_nom)
        polos = np.array(ct.poles(G_d))
        mag_max = max(np.abs(polos))
        imag_max = max(np.abs(np.imag(polos)))
        tem_oscilacao = imag_max > 1e-8
        estavel = mag_max < 1.0

        resultados.append({
            'K_nom': K_nom,
            'polos': polos,
            'mag_max': mag_max,
            'imag_max': imag_max,
            'oscila': tem_oscilacao,
            'estavel': estavel,
        })
    except:
        resultados.append({
            'K_nom': K_nom, 'polos': [], 'mag_max': float('nan'),
            'imag_max': float('nan'), 'oscila': None, 'estavel': None,
        })


# =============================================
# ENCONTRAR FRONTEIRAS
# =============================================

K_inicio_oscilacao = None
K_fim_oscilacao = None
K_instabilidade = None

for i in range(1, len(resultados)):
    rp, rc = resultados[i-1], resultados[i]
    if rp['oscila'] == False and rc['oscila'] == True and K_inicio_oscilacao is None:
        K_inicio_oscilacao = rc['K_nom']
    if rp['oscila'] == True and rc['oscila'] == False and K_fim_oscilacao is None:
        K_fim_oscilacao = rc['K_nom']

for r in resultados:
    if r['estavel'] == False:
        K_instabilidade = r['K_nom']
        break

print("\n" + "=" * 65)
print("  ANALISE COM Td FIXO — FRONTEIRAS DO SISTEMA")
print("=" * 65)

if K_inicio_oscilacao:
    print(f"\n  >> INICIO DA OSCILACAO:  K_nom = {K_inicio_oscilacao:.3f}")
else:
    print(f"\n  >> O sistema NAO OSCILA na faixa testada")

if K_fim_oscilacao:
    print(f"  >> FIM DA OSCILACAO:     K_nom = {K_fim_oscilacao:.3f}")

if K_instabilidade:
    print(f"  >> INICIO INSTABILIDADE: K_nom = {K_instabilidade:.3f}")
else:
    print(f"  >> O sistema permanece ESTAVEL na faixa testada")

estaveis_sem_osc = [r for r in resultados if r['estavel'] and not r['oscila']]
estaveis_com_osc = [r for r in resultados if r['estavel'] and r['oscila']]
instaveis = [r for r in resultados if r['estavel'] == False]

if estaveis_sem_osc:
    print(f"\n  FAIXA SEGURA (estavel, sem oscilacao):")
    print(f"     K_nom de {min(r['K_nom'] for r in estaveis_sem_osc):.3f} "
          f"a {max(r['K_nom'] for r in estaveis_sem_osc):.3f}")

if estaveis_com_osc:
    print(f"\n  FAIXA OSCILATORIA CONTROLADA (estavel, oscila):")
    print(f"     K_nom de {min(r['K_nom'] for r in estaveis_com_osc):.3f} "
          f"a {max(r['K_nom'] for r in estaveis_com_osc):.3f}")

if instaveis:
    print(f"\n  FAIXA INSTAVEL:")
    print(f"     K_nom a partir de {min(r['K_nom'] for r in instaveis):.3f}")

print(f"\n  Td fixo = {Td_fixo:.6f}  (projetado para K=30)")
print("=" * 65)


# =============================================
# GRÁFICO 1: Análise de Polos (3 subplots)
# =============================================

K_vals = [r['K_nom'] for r in resultados]
mag_vals = [r['mag_max'] for r in resultados]
imag_vals = [r['imag_max'] for r in resultados]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'Análise de Polos com Td FIXO = {Td_fixo:.4f} (projetado para K=30)',
             fontsize=14, fontweight='bold')

# Subplot 1: Magnitude
ax1 = axes[0]
colors1 = ['#2ca02c' if r['estavel'] else '#d62728' for r in resultados]
ax1.scatter(K_vals, mag_vals, c=colors1, s=8)
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
            label='|polo| = 1 (fronteira estabilidade)')
ax1.set_ylabel('|polo| máximo')
ax1.set_title('Magnitude máxima dos polos')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Parte imaginária
ax2 = axes[1]
colors2 = ['#ff7f0e' if r['oscila'] else '#2ca02c' for r in resultados]
ax2.scatter(K_vals, imag_vals, c=colors2, s=8)
ax2.set_ylabel('|Im(polo)| máximo')
ax2.set_title('Parte imaginária dos polos (> 0 indica oscilação)')
ax2.grid(True, alpha=0.3)

if K_inicio_oscilacao:
    ax2.axvline(x=K_inicio_oscilacao, color='red', linestyle=':', linewidth=2,
                label=f'Início oscilação: K = {K_inicio_oscilacao:.2f}')
    ax2.legend(fontsize=10)

# Subplot 3: Classificação
ax3 = axes[2]
status_vals = []
colors3 = []
for r in resultados:
    if r['estavel'] and not r['oscila']:
        status_vals.append(2); colors3.append('#2ca02c')
    elif r['estavel'] and r['oscila']:
        status_vals.append(1); colors3.append('#ff7f0e')
    elif not r['estavel']:
        status_vals.append(0); colors3.append('#d62728')
    else:
        status_vals.append(-1); colors3.append('#999999')

ax3.scatter(K_vals, status_vals, c=colors3, s=15)
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['Instável', 'Estável + Oscila', 'Estável + Sem Oscilação'])
ax3.set_xlabel('K_nom (γ)', fontsize=12)
ax3.set_title('Classificação do sistema')
ax3.grid(True, alpha=0.3)

for ax_i in [ax1, ax2, ax3]:
    if K_inicio_oscilacao:
        ax_i.axvline(x=K_inicio_oscilacao, color='#ff7f0e', linestyle=':', alpha=0.6)
    if K_instabilidade:
        ax_i.axvline(x=K_instabilidade, color='#d62728', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('analise_td_fixo_polos.png', dpi=150, bbox_inches='tight')
print("\nGrafico salvo: analise_td_fixo_polos.png")


# =============================================
# GRÁFICO 2: Respostas ao degrau selecionadas
# =============================================

# Escolhe valores representativos de cada faixa
K_exemplos = sorted(set([
    0.5, 1, 5, 10, 20, 30,
    K_inicio_oscilacao if K_inicio_oscilacao else 50,
    K_instabilidade * 0.5 if K_instabilidade else 100,
    K_instabilidade * 0.8 if K_instabilidade else 200,
    K_instabilidade * 0.95 if K_instabilidade else 300,
    K_instabilidade * 1.05 if K_instabilidade else 500,
    K_instabilidade * 1.5 if K_instabilidade else 1000,
]))

# Limitar a 12 para o grid 3x4
K_exemplos = K_exemplos[:12]

fig2, axes2 = plt.subplots(3, 4, figsize=(18, 12))
fig2.suptitle(f'Resposta ao Degrau com Td FIXO = {Td_fixo:.4f}', fontsize=14, fontweight='bold')

for idx, K_nom in enumerate(K_exemplos):
    ax = axes2[idx // 4, idx % 4]
    try:
        G_d = calcular_planta_td_fixo(K_nom)
        polos = np.array(ct.poles(G_d))
        mag_max = max(np.abs(polos))
        imag_max = max(np.abs(np.imag(polos)))

        t, y = ct.step_response(G_d, T=np.arange(0, 40, 1))

        oscila = imag_max > 1e-8
        estavel = mag_max < 1.0

        if estavel and not oscila:
            cor = '#2ca02c'; label = 'ESTÁVEL'
        elif estavel and oscila:
            cor = '#ff7f0e'; label = 'OSCILA (estável)'
        else:
            cor = '#d62728'; label = 'INSTÁVEL'

        ax.step(t, y, where='post', color=cor, linewidth=1.5)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'K = {K_nom:.1f}\n{label}', fontsize=9, color=cor, fontweight='bold')
        ax.set_xlabel('Tempo (s)', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.grid(True, alpha=0.3)

        if estavel:
            y_lim = max(2.0, max(np.abs(y)) * 1.2)
            ax.set_ylim([-y_lim, y_lim])

    except Exception as e:
        ax.text(0.5, 0.5, f'ERRO\n{str(e)[:40]}', transform=ax.transAxes,
                ha='center', va='center', color='red', fontsize=7)
        ax.set_title(f'K = {K_nom:.1f}', fontsize=9)

# Esconde subplots vazios se menos de 12 exemplos
for idx in range(len(K_exemplos), 12):
    axes2[idx // 4, idx % 4].set_visible(False)

plt.tight_layout()
plt.savefig('respostas_td_fixo.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: respostas_td_fixo.png")


# =============================================
# GRÁFICO 3: Mapa de polos no plano Z
# =============================================

K_polos = sorted(set([
    1, 10, 30,
    K_inicio_oscilacao if K_inicio_oscilacao else 50,
    K_instabilidade * 0.5 if K_instabilidade else 100,
    K_instabilidade * 0.9 if K_instabilidade else 200,
    K_instabilidade * 1.0 if K_instabilidade else 400,
    K_instabilidade * 1.1 if K_instabilidade else 600,
    K_instabilidade * 2.0 if K_instabilidade else 1000,
    K_instabilidade * 5.0 if K_instabilidade else 3000,
]))[:10]

fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
fig3.suptitle('Polos no Plano Z — Td Fixo', fontsize=14, fontweight='bold')

for idx, K_nom in enumerate(K_polos):
    ax = axes3[idx // 5, idx % 5]
    try:
        G_d = calcular_planta_td_fixo(K_nom)
        polos = np.array(ct.poles(G_d))

        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)

        for p in polos:
            cor = '#d62728' if abs(p) >= 1 else ('#ff7f0e' if abs(np.imag(p)) > 1e-8 else '#2ca02c')
            ax.plot(np.real(p), np.imag(p), 'x', color=cor, markersize=12, markeredgewidth=2)

        oscila_str = "OSCILA" if max(abs(np.imag(polos))) > 1e-8 else "SEM OSC."
        estavel_str = "EST." if max(abs(polos)) < 1 else "INST."
        cor_titulo = '#2ca02c' if max(abs(polos)) < 1 else '#d62728'

        ax.set_title(f'K={K_nom:.0f}\n{estavel_str}, {oscila_str}', fontsize=8, color=cor_titulo)
        lim = max(1.5, max(np.abs(polos)) * 1.2)
        ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
    except:
        ax.text(0.5, 0.5, 'ERRO', transform=ax.transAxes, ha='center')

for idx in range(len(K_polos), 10):
    axes3[idx // 5, idx % 5].set_visible(False)

plt.tight_layout()
plt.savefig('polos_plano_z_td_fixo.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: polos_plano_z_td_fixo.png")
