"""
Varredura empírica de K_nom (gamma) para encontrar a faixa de estabilidade.

Gera dois gráficos:
1. Várias respostas ao degrau sobrepostas para diferentes K_nom
2. Um mapa de estabilidade mostrando quais valores de K_nom são estáveis/instáveis
"""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo — salva PNG sem travar
import control as ct
import math as m
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros nominais (do artigo)
a_n = 1.6
b_n = 0.2
un = 1  # 1 m/s

s = ct.tf('s')


def calcular_planta_nominal(K_nom):
    """
    Calcula a planta discretizada para velocidade nominal (u = un)
    com o K_nom (gamma) fornecido.
    """
    eta = 1.0  # u/un = 1 para caso nominal
    a = a_n * eta      # = 1.6
    b = b_n * eta**2   # = 0.2
    K = K_nom / eta**2  # = K_nom (pois eta=1)

    # Td adaptativo (xi = 2.4)
    Td = (2.4 * m.sqrt(0.2 * K) - 1.6) / (0.2 * K)

    # Função de transferência em malha fechada com PD
    G = (K * b * (1 + s * Td)) / (s**2 + s * (a + K * b * Td) + K * b)

    # Discretiza com período de amostragem = 1s
    G_discreto = ct.c2d(G, 1)

    return G_discreto


def classificar_resposta(t, y):
    """
    Classifica a resposta ao degrau como:
    - 'estavel'     : converge sem oscilar demais
    - 'oscilatorio' : oscila mas não diverge
    - 'instavel'    : diverge ou oscila crescentemente
    """
    # Verifica divergência
    if np.any(np.abs(y) > 1e3) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return 'instavel'

    # Conta cruzamentos de zero (indicador de oscilação)
    y_centered = y - y[-1]  # centra no valor final
    sign_changes = np.sum(np.diff(np.sign(y_centered)) != 0)

    # Verifica se o valor final está próximo de 1.0 (degrau unitário)
    valor_final = y[-1]

    # Verifica se amplitude das oscilações está crescendo
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            peaks.append(abs(y[i] - valor_final))

    if len(peaks) >= 3:
        # Se os picos crescem, é instável
        if peaks[-1] > peaks[0] * 1.5:
            return 'instavel'

    if sign_changes > 6:
        return 'oscilatorio'

    if abs(valor_final - 1.0) > 0.5:
        return 'divergente'

    return 'estavel'


# =============================================
# GRÁFICO 1: Respostas ao degrau sobrepostas
# =============================================

# Valores de K_nom para testar (escala logarítmica para cobrir ampla faixa)
K_values_plot = [0.01, 0.1, 1, 5, 10, 20, 30, 50, 80, 100, 150, 200]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('Resposta ao Degrau para diferentes valores de K_nom (gamma)', fontsize=14, fontweight='bold')

for idx, K_nom in enumerate(K_values_plot):
    ax = axes[idx // 4, idx % 4]
    try:
        G_d = calcular_planta_nominal(K_nom)
        t, y = ct.step_response(G_d, T=np.arange(0, 30, 1))
        status = classificar_resposta(t, y)

        color = {'estavel': '#2ca02c', 'oscilatorio': '#ff7f0e',
                 'instavel': '#d62728', 'divergente': '#d62728'}[status]

        ax.step(t, y, where='post', color=color, linewidth=1.5)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ref = 1.0')
        ax.set_title(f'K_nom = {K_nom}\n({status.upper()})', fontsize=10,
                     color=color, fontweight='bold')
        ax.set_xlabel('Tempo (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Limita eixo Y para visualização
        y_max = min(max(np.abs(y)) * 1.3, 5)
        ax.set_ylim([-y_max, y_max])

    except Exception as e:
        ax.text(0.5, 0.5, f'ERRO\n{str(e)[:30]}', transform=ax.transAxes,
                ha='center', va='center', color='red', fontsize=8)
        ax.set_title(f'K_nom = {K_nom}', fontsize=10)

plt.tight_layout()
plt.savefig('varredura_degraus.png', dpi=150, bbox_inches='tight')
print("Grafico 1 salvo: varredura_degraus.png")

# =============================================
# GRÁFICO 2: Mapa de estabilidade
# =============================================

K_sweep = np.concatenate([
    np.arange(0.01, 1, 0.05),
    np.arange(1, 10, 0.5),
    np.arange(10, 50, 1),
    np.arange(50, 300, 5)
])

resultados = []

for K_nom in K_sweep:
    try:
        G_d = calcular_planta_nominal(K_nom)
        t, y = ct.step_response(G_d, T=np.arange(0, 50, 1))
        status = classificar_resposta(t, y)

        # Métrica de overshoot (quanto ultrapassa o valor final)
        valor_final = y[-1]
        if abs(valor_final) > 1e-6:
            overshoot = (max(y) - valor_final) / abs(valor_final) * 100
        else:
            overshoot = float('inf')

        # Tempo de acomodação (settling time)
        settling_time = len(t)
        for i in range(len(y) - 1, -1, -1):
            if abs(y[i] - valor_final) > 0.05 * abs(valor_final + 1e-10):
                settling_time = t[i] if i < len(t) else len(t)
                break

        resultados.append({
            'K_nom': K_nom,
            'status': status,
            'overshoot': overshoot,
            'valor_final': valor_final,
            'settling_time': settling_time
        })
    except:
        resultados.append({
            'K_nom': K_nom,
            'status': 'erro',
            'overshoot': float('nan'),
            'valor_final': float('nan'),
            'settling_time': float('nan')
        })

# Plota o mapa de estabilidade
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle('Mapa de Estabilidade vs K_nom (gamma)', fontsize=14, fontweight='bold')

K_vals = [r['K_nom'] for r in resultados]
overshoots = [r['overshoot'] for r in resultados]
valores_finais = [r['valor_final'] for r in resultados]
settling_times = [r['settling_time'] for r in resultados]
statuses = [r['status'] for r in resultados]

# Cores por status
colors = []
for s_val in statuses:
    if s_val == 'estavel':
        colors.append('#2ca02c')
    elif s_val == 'oscilatorio':
        colors.append('#ff7f0e')
    elif s_val in ('instavel', 'divergente'):
        colors.append('#d62728')
    else:
        colors.append('#999999')

# Subplot 1: Valor final
ax1.scatter(K_vals, valores_finais, c=colors, s=10)
ax1.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Referencia = 1.0')
ax1.set_ylabel('Valor Final')
ax1.set_title('Valor de regime permanente')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Overshoot
ax2.scatter(K_vals, np.clip(overshoots, -100, 500), c=colors, s=10)
ax2.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
ax2.set_ylabel('Overshoot (%)')
ax2.set_title('Sobressinal percentual')
ax2.grid(True, alpha=0.3)

# Subplot 3: Status como faixa colorida
status_map = {'estavel': 1, 'oscilatorio': 0.5, 'divergente': 0, 'instavel': -0.5, 'erro': -1}
status_nums = [status_map.get(s_val, -1) for s_val in statuses]
ax3.scatter(K_vals, status_nums, c=colors, s=20)
ax3.set_yticks([1, 0.5, 0, -0.5])
ax3.set_yticklabels(['Estavel', 'Oscilatorio', 'Divergente', 'Instavel'])
ax3.set_xlabel('K_nom (gamma)')
ax3.set_title('Classificacao de estabilidade')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mapa_estabilidade.png', dpi=150, bbox_inches='tight')
print("Grafico 2 salvo: mapa_estabilidade.png")

# =============================================
# RESUMO NO TERMINAL
# =============================================

print("\n" + "="*60)
print("RESUMO DA VARREDURA DE K_nom (gamma)")
print("="*60)

estaveis = [r for r in resultados if r['status'] == 'estavel']
oscilatorios = [r for r in resultados if r['status'] == 'oscilatorio']
instaveis = [r for r in resultados if r['status'] in ('instavel', 'divergente')]

if estaveis:
    k_min_estavel = min(r['K_nom'] for r in estaveis)
    k_max_estavel = max(r['K_nom'] for r in estaveis)
    print(f"\nFAIXA ESTAVEL: K_nom de {k_min_estavel:.2f} a {k_max_estavel:.2f}")
else:
    print("\nNenhum valor de K_nom resultou em sistema estavel!")

if oscilatorios:
    print(f"OSCILATORIOS: K_nom de {min(r['K_nom'] for r in oscilatorios):.2f} "
          f"a {max(r['K_nom'] for r in oscilatorios):.2f}")

if instaveis:
    print(f"INSTAVEIS: K_nom de {min(r['K_nom'] for r in instaveis):.2f} "
          f"a {max(r['K_nom'] for r in instaveis):.2f}")

print(f"\nValor original do MATLAB: K = 30")
print(f"Valor atual no seu codigo: K_nom = 0.01")
print("="*60)

