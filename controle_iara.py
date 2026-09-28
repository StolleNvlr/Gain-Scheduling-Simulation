import control as ct
import math as m
import matplotlib.pyplot as plt

# ====================================================================
# PARTE 1: DEFINIÇÃO MATEMÁTICA E VALIDAÇÃO ACADÊMICA
# ====================================================================

# 1. Parâmetros Nominais
a_n = 1.6
b_n = 0.2
un = 1 # 1 m/s nominal pós conversão de nós para m/s
s = ct.tf('s')

def calcular_planta(velocidade_alvo):
    """
    Realiza o Gain Scheduling: Recalcula a Função de Transferência
    ajustando os ganhos PD (K e Td) para a velocidade exigida.
    """
    # Proporção da velocidade (eta)
    eta = velocidade_alvo / un
    a = a_n * eta
    b = b_n * (eta ** 2)
    
    # O Ganho Nominal projetado foi K = 30
    K_nom = 30
    K = K_nom / (eta ** 2)
    
    # Cálculo adaptativo de Td conforme o artigo (para xi = 2.4)
    if velocidade_alvo == 0.8 * un:
        Td = (2.4 / 0.8 * m.sqrt(0.128 * K) - 1.28) / (0.128 * K)
    elif velocidade_alvo == 0.1 * un:
        Td = (2.4 / 0.1 * m.sqrt(2e-3 * K) - 0.16) / (2e-3 * K)
    else: # velocidade nominal (1.0)
        Td = (2.4 * m.sqrt(0.2 * K) - 1.6) / (0.2 * K)

    # G(s) - Controlador PD aplicado à Planta
    G = (K * b * (1 + s * Td)) / (s**2 + s * (a + K * b * Td) + K * b)
    
    # Discretiza a planta usando Amostragem de 1s (Conforme o original)
    G_discreto = ct.c2d(G, 1)
    
    return G, G_discreto

# Gera os 3 sistemas discretos definidos no trabalho
_, G_nom = calcular_planta(1.0)
_, G_08 = calcular_planta(0.8)
_, G_01 = calcular_planta(0.1)

def plotar_degraus():
    """Gera os gráficos das Figuras 3, 4 e 5 do trabalho original em janelas separadas"""
    
    # ==========================================
    # Figura 3: Caso Nominal (u = un)
    # ==========================================
    plt.figure(figsize=(8, 5))
    t, y_nom = ct.step_response(G_nom)
    plt.step(t, y_nom, label="Caso u = un", where='post', color='#1f77b4')
    plt.title('Resposta ao Degrau - Caso u = un (Figura 3)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # ==========================================
    # Figura 4: Caso 0.8 un
    # ==========================================
    plt.figure(figsize=(8, 5))
    t, y_08 = ct.step_response(G_08)
    plt.step(t, y_08 * m.atan(10/7), label="Caso u = 0.8 un", where='post', color='#ff7f0e')
    plt.title('Resposta ao Degrau - Caso u = 0.8 un (Figura 4)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # ==========================================
    # Figura 5: Caso 0.1 un
    # ==========================================
    plt.figure(figsize=(8, 5))
    t, y_01 = ct.step_response(G_01)
    plt.step(t, y_01, label="Caso u = 0.1 un", where='post', color='#2ca02c')
    plt.title('Resposta ao Degrau - Caso u = 0.1 un (Figura 5)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # O comando show() no final vai abrir as três janelas simultaneamente
    plt.show()


# ====================================================================
# PARTE 2: MOTOR DE CONTROLE CINEMÁTICO (USADO NO RUNNER.PY)
# ====================================================================

class ControladorIara:
    """
    Objeto que retém a memória dos regressores e aplica o filtro
    das equações a diferenças para navegação em tempo real.
    """
    def __init__(self):
        # Memória das entradas e saídas passadas (k-1, k-2)
        self.i1 = 0.0 
        self.i2 = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        
    def atualizar_angulo(self, estado_velocidade, i_ang):
        """
        Calcula a rotação suave do leme (s_ang) baseada na 
        velocidade atual e na direção do próximo waypoint (i_ang).
        """
        # Aplica a equação a diferenças extraída da transformada Z
        if estado_velocidade == 0.8:
            s_ang = 0.279 * self.s1 - 0.002798 * self.s2 + 0.9669 * self.i1 - 0.2431 * self.i2
        elif estado_velocidade == 0.1:
            s_ang = 0.394 * self.s1 - 0.0006436 * self.s2 + 0.9738 * self.i1 - 0.3671 * self.i2
        else: # Estado = 1.0
            s_ang = 0.9028 * self.s1 - 1.002e-16 * self.s2 + 0.9991 * self.i1 - 0.9019 * self.i2
            
        # Desloca a memória para o próximo ciclo do CoppeliaSim
        self.i2 = self.i1
        self.i1 = i_ang
        self.s2 = self.s1
        self.s1 = s_ang
        
        return s_ang
if __name__ == '__main__':
    plotar_degraus()
