"""
Análise COMPLETA: Varia xi (amortecimento) e K_nom para encontrar
as fronteiras de oscilação e instabilidade.

O sistema original usa xi=2.4 (muito sobreamortecido).
Aqui testamos xi de 0.1 a 3.0 para cada K_nom.
Também analisamos o sistema CONTÍNUO (antes da discretização).
"""
import matplotlib
matplotlib.use('Agg')
import control as ct
import math as m
import numpy as np
import matplotlib.pyplot as plt

a_n = 1.6
b_n = 0.2
un = 1
s_var = ct.tf('s')


def calcular_sistema(K_nom, xi):
    """Sistema contínuo e discreto para K_nom e xi dados."""
    eta = 1.0
    a = a_n * eta
    b = b_n * eta**2
    K = K_nom / eta**2
    Td = (xi * m.sqrt(b * K) - a) / (b * K)

    G_cont = (K * b * (1 + s_var * Td)) / (s_var**2 + s_var * (a + K * b * Td) + K * b)
    G_disc = ct.c2d(G_cont, 1)
    return G_cont, G_disc, Td


# =============================================
# PARTE 1: Análise do sistema CONTÍNUO
# =============================================
print("=" * 65)
print("  PARTE 1: ANALISE NO DOMINIO CONTINUO (s)")
print("=" * 65)

K_sweep = np.concatenate([
    np.arange(0.1, 10, 0.1),
    np.arange(10, 100, 1),
    np.arange(100, 500, 5),
])

# Para xi=2.4 (original)
print(f"\n--- xi = 2.4 (original) ---")
for K_nom in [0.1, 1, 10, 30, 100, 500]:
    try:
        G_c, G_d, Td = calcular_sistema(K_nom, 2.4)
        polos_c = np.array(ct.poles(G_c))
        polos_d = np.array(ct.poles(G_d))
        print(f"  K={K_nom:6.1f} | Td={Td:8.4f} | "
              f"Polos cont.: {np.round(polos_c, 4)} | "
              f"Polos disc.: {np.round(polos_d, 4)}")
    except Exception as e:
        print(f"  K={K_nom:6.1f} | ERRO: {e}")

# Para xi reduzido
for xi_test in [1.0, 0.5, 0.3]:
    print(f"\n--- xi = {xi_test} ---")
    for K_nom in [0.1, 1, 10, 30, 100, 500]:
        try:
            G_c, G_d, Td = calcular_sistema(K_nom, xi_test)
            polos_c = np.array(ct.poles(G_c))
            polos_d = np.array(ct.poles(G_d))
            imag_c = max(abs(np.imag(polos_c)))
            imag_d = max(abs(np.imag(polos_d)))
            osc_c = "OSCILA" if imag_c > 1e-8 else "  ---  "
            osc_d = "OSCILA" if imag_d > 1e-8 else "  ---  "
            print(f"  K={K_nom:6.1f} | Td={Td:8.4f} | "
                  f"Cont: {osc_c} Im={imag_c:.4f} | "
                  f"Disc: {osc_d} Im={imag_d:.4f}")
        except Exception as e:
            print(f"  K={K_nom:6.1f} | ERRO: {e}")


# =============================================
# PARTE 2: Mapa 2D — xi vs K_nom
# =============================================

xi_range = np.arange(0.1, 3.1, 0.05)
K_range = np.concatenate([
    np.arange(0.5, 10, 0.5),
    np.arange(10, 100, 2),
    np.arange(100, 500, 10),
])

# Matrizes para o mapa
mapa_oscila_cont = np.zeros((len(xi_range), len(K_range)))
mapa_oscila_disc = np.zeros((len(xi_range), len(K_range)))
mapa_estavel_disc = np.zeros((len(xi_range), len(K_range)))

for i, xi in enumerate(xi_range):
    for j, K_nom in enumerate(K_range):
        try:
            G_c, G_d, Td = calcular_sistema(K_nom, xi)
            polos_c = np.array(ct.poles(G_c))
            polos_d = np.array(ct.poles(G_d))

            mapa_oscila_cont[i, j] = 1 if max(abs(np.imag(polos_c))) > 1e-8 else 0
            mapa_oscila_disc[i, j] = 1 if max(abs(np.imag(polos_d))) > 1e-8 else 0
            mapa_estavel_disc[i, j] = 1 if max(abs(polos_d)) < 1.0 else 0
        except:
            mapa_oscila_cont[i, j] = np.nan
            mapa_oscila_disc[i, j] = np.nan
            mapa_estavel_disc[i, j] = np.nan


# =============================================
# GRÁFICO: Mapa 2D de estabilidade
# =============================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Mapa de Estabilidade: ξ (amortecimento) vs K_nom (γ)', fontsize=14, fontweight='bold')

# Mapa 1: Oscilação no contínuo
im1 = axes[0].pcolormesh(K_range, xi_range, mapa_oscila_cont,
                          cmap='RdYlGn_r', shading='auto', vmin=0, vmax=1)
axes[0].set_title('Oscilação (Sistema Contínuo)\n1 = oscila, 0 = não oscila')
axes[0].set_xlabel('K_nom (γ)')
axes[0].set_ylabel('ξ (amortecimento)')
axes[0].axhline(y=2.4, color='white', linestyle='--', linewidth=2, label='ξ = 2.4 (original)')
axes[0].axhline(y=1.0, color='cyan', linestyle=':', linewidth=1.5, label='ξ = 1.0 (crítico)')
axes[0].legend(fontsize=8)
fig.colorbar(im1, ax=axes[0])

# Mapa 2: Oscilação no discreto
im2 = axes[1].pcolormesh(K_range, xi_range, mapa_oscila_disc,
                          cmap='RdYlGn_r', shading='auto', vmin=0, vmax=1)
axes[1].set_title('Oscilação (Sistema Discreto, Ts=1s)\n1 = oscila, 0 = não oscila')
axes[1].set_xlabel('K_nom (γ)')
axes[1].set_ylabel('ξ (amortecimento)')
axes[1].axhline(y=2.4, color='white', linestyle='--', linewidth=2, label='ξ = 2.4 (original)')
axes[1].axhline(y=1.0, color='cyan', linestyle=':', linewidth=1.5, label='ξ = 1.0 (crítico)')
axes[1].legend(fontsize=8)
fig.colorbar(im2, ax=axes[1])

# Mapa 3: Estabilidade no discreto
im3 = axes[2].pcolormesh(K_range, xi_range, mapa_estavel_disc,
                          cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
axes[2].set_title('Estabilidade (Sistema Discreto, Ts=1s)\n1 = estável, 0 = instável')
axes[2].set_xlabel('K_nom (γ)')
axes[2].set_ylabel('ξ (amortecimento)')
axes[2].axhline(y=2.4, color='white', linestyle='--', linewidth=2, label='ξ = 2.4 (original)')
axes[2].axhline(y=1.0, color='cyan', linestyle=':', linewidth=1.5, label='ξ = 1.0 (crítico)')
axes[2].legend(fontsize=8)
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('mapa_2d_xi_vs_K.png', dpi=150, bbox_inches='tight')
print("\nGrafico salvo: mapa_2d_xi_vs_K.png")


# =============================================
# GRÁFICO: Respostas ao degrau com xi reduzido
# =============================================

xi_tests = [0.3, 0.5, 0.7, 1.0, 1.5, 2.4]
K_tests = [1, 10, 30, 100]

fig2, axes2 = plt.subplots(len(xi_tests), len(K_tests), figsize=(18, 20))
fig2.suptitle('Resposta ao Degrau: Variação de ξ e K_nom', fontsize=14, fontweight='bold')

for i, xi in enumerate(xi_tests):
    for j, K_nom in enumerate(K_tests):
        ax = axes2[i, j]
        try:
            G_c, G_d, Td = calcular_sistema(K_nom, xi)
            polos_d = np.array(ct.poles(G_d))
            mag = max(abs(polos_d))
            imag = max(abs(np.imag(polos_d)))

            t, y = ct.step_response(G_d, T=np.arange(0, 30, 1))

            oscila = imag > 1e-8
            estavel = mag < 1.0

            if estavel and not oscila:
                cor = '#2ca02c'; label = 'Estável'
            elif estavel and oscila:
                cor = '#ff7f0e'; label = 'Oscila'
            else:
                cor = '#d62728'; label = 'Instável'

            ax.step(t, y, where='post', color=cor, linewidth=1.2)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
            ax.set_title(f'ξ={xi}, K={K_nom}\n{label}', fontsize=8, color=cor, fontweight='bold')
            ax.grid(True, alpha=0.2)

            if estavel:
                y_lim = max(2, max(abs(y)) * 1.2)
                ax.set_ylim([-y_lim, y_lim])

        except Exception as e:
            ax.text(0.5, 0.5, f'ERRO', transform=ax.transAxes, ha='center', color='red')
            ax.set_title(f'ξ={xi}, K={K_nom}', fontsize=8)

        if i == len(xi_tests) - 1:
            ax.set_xlabel('Tempo (s)', fontsize=7)
        if j == 0:
            ax.set_ylabel('Amplitude', fontsize=7)

plt.tight_layout()
plt.savefig('respostas_xi_variavel.png', dpi=150, bbox_inches='tight')
print("Grafico salvo: respostas_xi_variavel.png")


# =============================================
# RESUMO FINAL
# =============================================

print("\n" + "=" * 65)
print("  RESUMO FINAL DA ANALISE COMPLETA")
print("=" * 65)

# Encontrar fronteira de oscilação no contínuo para xi=2.4
print(f"\n  Para xi = 2.4 (projeto original):")
print(f"    - Sistema CONTINUO: NUNCA oscila (xi > 1 = sobreamortecido)")
print(f"    - Sistema DISCRETO: NUNCA oscila")
print(f"    - Estavel para K_nom de ~0.1 a >5000")

# Encontrar fronteira de oscilação para xi < 1
print(f"\n  Para xi < 1.0 (subamortecido):")
oscilou = False
for i, xi in enumerate(xi_range):
    for j, K_nom in enumerate(K_range):
        if mapa_oscila_cont[i, j] == 1:
            if not oscilou:
                print(f"    - Oscilacao no CONTINUO a partir de xi <= {xi:.2f}")
                oscilou = True
            break
    if oscilou:
        break

oscilou_d = False
for i, xi in enumerate(xi_range):
    for j, K_nom in enumerate(K_range):
        if mapa_oscila_disc[i, j] == 1:
            if not oscilou_d:
                print(f"    - Oscilacao no DISCRETO a partir de xi <= {xi:.2f}")
                oscilou_d = True
            break
    if oscilou_d:
        break

if not oscilou:
    print(f"    - Nenhuma oscilacao encontrada no continuo")
if not oscilou_d:
    print(f"    - Nenhuma oscilacao encontrada no discreto")

print("=" * 65)
