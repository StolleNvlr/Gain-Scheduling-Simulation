from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math as m
import controle_iara
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
client = RemoteAPIClient()
sim = client.require('sim')
controlador = controle_iara.ControladorIara()
'''
variaveis globais para o funcionamento do RRT*
'''
X_MIN = -0.5
X_MAX = 4.5
Y_MIN = -1.2
Y_MAX = 1.5

PASSO = 0.4
MAX_ITERACOES = 5000
GOAL_BIAS = 0.10
RAIO_REWIRE = 1.0  # Raio de busca para reconexão de vizinhos no RRT*

nomes = ['Cuboid_1','Cuboid_2','Cuboid_3']
barco = sim.getObject('/iara')

client.setStepping(True)
sim.setFloatParam(sim.floatparam_simulation_time_step, 0.05)
'''
Path Planning em RRT
'''
def distancia(a,b):
    return m.hypot(a[0]-b[0], a[1]-b[1])

def input_de_ponto():
    texto = input("Insira os waypoints separados por ';' (Ex: 2.0,1.0 ; 3.0,-0.7) : ")
    lista_str = texto.split(';')
    pontos = []
    for p in lista_str:
        coords = p.split(',')
        # O strip() remove espaços em branco antes de converter para float
        pontos.append((float(coords[0].strip()), float(coords[1].strip())))
        
    return pontos

def posicao(barco_handle):
    posicao_inicial = sim.getObjectPosition(barco_handle, sim.handle_world)
    return posicao_inicial
def obstaculos_cena(nome_obstaculos, margem = 0.0):
    lista_obstaculos = []
    for nome in nome_obstaculos:
        try:
            handle = sim.getObject(f'/{nome}')
            pos = sim.getObjectPosition(handle, sim.handle_world)
            
            # Lê os limites extremos da malha 3D do objeto no CoppeliaSim (Seguro contra erros de API)
            min_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
            max_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x)
            min_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
            max_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y)
            
            # Calcula o tamanho exato e soma a margem de segurança do barco
            largura = (max_x - min_x) + margem
            profundidade = (max_y - min_y) + margem
            
            # Agora salvamos (X_Centro, Y_Centro, Largura, Profundidade)
            lista_obstaculos.append((pos[0], pos[1], largura, profundidade))
            print(f"Obstáculo '{nome}' carregado: X={pos[0]:.2f}, Y={pos[1]:.2f}, Larg={largura:.2f}, Prof={profundidade:.2f}")
            
        except Exception as e:
            print(f"Aviso: Não carregou '{nome}'. Erro: {e}")
            
    return lista_obstaculos

def ponto_aleatorio():
    return (random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX))

def ponto_valido(p, obstaculos):
    x, y = p
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
        return False
        
    for ox, oy, larg, prof in obstaculos:
        # Define as quatro paredes da zona de colisão
        x_min = ox - (larg / 2.0)
        x_max = ox + (larg / 2.0)
        y_min = oy - (prof / 2.0)
        y_max = oy + (prof / 2.0)
        
        # Se o ponto sorteado cair dentro desse quadrado, é inválido
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return False
            
    return True

def colide_segmento(a, b, obstaculos, resolucao=0.03):
    comprimento = distancia(a, b)
    passos = max(1, int(comprimento / resolucao))
    for i in range(passos + 1):
        t = i / passos
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if not ponto_valido((x, y), obstaculos):
            return True
    return False

def no_mais_proximo(arvore, ponto):
    return min(range(len(arvore)), key=lambda i: distancia(arvore[i]["ponto"], ponto))

def steer(origem, destino, passo):
    d = distancia(origem, destino)
    if d == 0: return origem
    if d <= passo: return destino
    dx, dy = (destino[0] - origem[0]) / d, (destino[1] - origem[1]) / d
    return (origem[0] + passo * dx, origem[1] + passo * dy)

def reconstruir_caminho(arvore, indice_final):
    caminho = []
    while indice_final is not None:
        caminho.append(arvore[indice_final]["ponto"])
        indice_final = arvore[indice_final]["pai"]
    return list(reversed(caminho))

def _propagar_custo(arvore, indice):
    """Propaga a atualização de custo para todos os descendentes de um nó reconectado."""
    for i in range(len(arvore)):
        if arvore[i]["pai"] == indice:
            arvore[i]["custo"] = arvore[indice]["custo"] + distancia(arvore[indice]["ponto"], arvore[i]["ponto"])
            _propagar_custo(arvore, i)

def RRT_Star(posicao_inicial, objetivo, obstaculos):
    if not ponto_valido(posicao_inicial, obstaculos) or not ponto_valido(objetivo, obstaculos):
        print("Erro: Ponto inicial ou objetivo estão dentro de um obstáculo ou fora dos limites!")
        return None, None

    # Cada nó agora guarda o custo acumulado desde a raiz (cache em O(1))
    arvore = [{"ponto": posicao_inicial, "pai": None, "custo": 0.0}]
    indice_objetivo = None
    melhor_custo = float('inf')
    historico_melhores_caminhos = []

    for iteracao in range(MAX_ITERACOES):
        # Amostragem com bias para o objetivo
        q_rand = objetivo if random.random() < GOAL_BIAS else ponto_aleatorio()

        # =========================================================
        # KD-Tree: Constrói a árvore espacial para buscas em O(log N)
        # =========================================================
        pontos_arvore = [no["ponto"] for no in arvore]
        kdtree = KDTree(pontos_arvore)

        # Nó mais próximo via KDTree (substitui busca linear)
        _, indice_near = kdtree.query(q_rand)
        q_near = arvore[indice_near]["ponto"]
        q_new = steer(q_near, q_rand, PASSO)

        # Evita adicionar nós duplicados (distância zero)
        if distancia(q_near, q_new) < 1e-5:
            continue

        if not ponto_valido(q_new, obstaculos) or colide_segmento(q_near, q_new, obstaculos):
            continue

        # =========================================================
        # PASSO 1 do RRT*: Escolher o melhor pai dentre os vizinhos
        # Busca todos os vizinhos no raio via KDTree (O(log N + k))
        # e escolhe o que resulta no menor custo acumulado.
        # =========================================================
        indices_vizinhos = kdtree.query_ball_point(q_new, RAIO_REWIRE)
        melhor_pai = indice_near
        melhor_custo_novo = arvore[indice_near]["custo"] + distancia(q_near, q_new)

        for idx_viz in indices_vizinhos:
            custo_candidato = arvore[idx_viz]["custo"] + distancia(arvore[idx_viz]["ponto"], q_new)
            if custo_candidato < melhor_custo_novo:
                if not colide_segmento(arvore[idx_viz]["ponto"], q_new, obstaculos):
                    melhor_pai = idx_viz
                    melhor_custo_novo = custo_candidato

        arvore.append({"ponto": q_new, "pai": melhor_pai, "custo": melhor_custo_novo})
        indice_new = len(arvore) - 1

        # =========================================================
        # PASSO 2 do RRT*: Rewiring (Reconexão da vizinhança)
        # Para cada vizinho, verifica se passar pelo novo nó q_new
        # seria mais barato. Se sim, reconecta e propaga o custo
        # atualizado para todos os descendentes.
        # =========================================================
        for idx_viz in indices_vizinhos:
            if idx_viz == melhor_pai:
                continue
            custo_via_novo = melhor_custo_novo + distancia(q_new, arvore[idx_viz]["ponto"])
            if custo_via_novo < arvore[idx_viz]["custo"]:
                if not colide_segmento(q_new, arvore[idx_viz]["ponto"], obstaculos):
                    # Evita ciclos: verifica se idx_viz é ancestral de indice_new
                    curr = indice_new
                    ciclo = False
                    while curr is not None:
                        if curr == idx_viz:
                            ciclo = True
                            break
                        curr = arvore[curr]["pai"]
                    
                    if not ciclo:
                        arvore[idx_viz]["pai"] = indice_new
                        arvore[idx_viz]["custo"] = custo_via_novo
                        _propagar_custo(arvore, idx_viz)  # Atualiza custos dos descendentes

        # =========================================================
        # PASSO 3: Verificar se alcançou o objetivo
        # Diferente do RRT puro, o RRT* NÃO para aqui. Ele registra
        # o caminho e continua otimizando até esgotar as iterações.
        # =========================================================
        if distancia(q_new, objetivo) <= PASSO and not colide_segmento(q_new, objetivo, obstaculos):
            custo_chegada = melhor_custo_novo + distancia(q_new, objetivo)
            if custo_chegada < melhor_custo:
                if indice_objetivo is not None:
                    # Evita ciclos: verifica se indice_objetivo é ancestral de indice_new
                    curr = indice_new
                    ciclo = False
                    while curr is not None:
                        if curr == indice_objetivo:
                            ciclo = True
                            break
                        curr = arvore[curr]["pai"]
                        
                    if not ciclo:
                        arvore[indice_objetivo]["pai"] = indice_new
                        arvore[indice_objetivo]["custo"] = custo_chegada
                        melhor_custo = custo_chegada
                        _propagar_custo(arvore, indice_objetivo)
                else:
                    arvore.append({"ponto": objetivo, "pai": indice_new, "custo": custo_chegada})
                    indice_objetivo = len(arvore) - 1
                    melhor_custo = custo_chegada
                
                caminho_atual = reconstruir_caminho(arvore, indice_objetivo)
                historico_melhores_caminhos.append({
                    "iteracao": iteracao,
                    "custo": melhor_custo,
                    "caminho": caminho_atual
                })
                
                print(f"  RRT* encontrou caminho (custo={melhor_custo:.2f}) na iteração {iteracao}. Continuando a otimizar...")
                print(f"    -> Pontos do caminho atual: {caminho_atual}")

    # Resultado final
    if indice_objetivo is not None:
        print(f"RRT* finalizado! Melhor caminho com custo={melhor_custo:.2f} e {len(arvore)} nós na árvore.")
        return reconstruir_caminho(arvore, indice_objetivo), arvore, historico_melhores_caminhos
    else:
        print("RRT* não encontrou caminho.")
        return None, arvore, []
def visualizar_rrt_completo(dados_plot, obstaculos):
    plt.figure(figsize=(10, 10))
    import matplotlib.patches as patches
    
    # 1. Desenha o mapa base (obstáculos retangulares e pontos) com zorder=1 (Fundo)
    for ox, oy, larg, prof in obstaculos:
        canto_x = ox - (larg / 2.0)
        canto_y = oy - (prof / 2.0)
        retangulo = patches.Rectangle((canto_x, canto_y), larg, prof, color='black', alpha=0.3, zorder=1)
        plt.gca().add_patch(retangulo)
        
    plt.title('Rota Completa Otimizada (RRT*)')
    plt.xlabel('Eixo X (metros)')
    plt.ylabel('Eixo Y (metros)')
    plt.grid(True)
    plt.axis('equal')
    
    # ==========================================
    # Renderização da árvore de caminhos de uma vez só (zorder=2)
    # ==========================================
    for arvore, _, _, _ in dados_plot:
        for no in arvore:
            if no["pai"] is not None:
                ponto_atual = no["ponto"]
                ponto_pai = arvore[no["pai"]]["ponto"]
                plt.plot([ponto_pai[0], ponto_atual[0]], [ponto_pai[1], ponto_atual[1]], 
                         color='cyan', alpha=0.3, linewidth=1, zorder=2)
                         
    # ==========================================
    # Desenha os pontos de Início e Objetivo no topo das linhas azuis (zorder=3)
    # ==========================================
    for i, (_, _, inicio, objetivo) in enumerate(dados_plot):
        label_inicio = 'Início' if i == 0 else ""
        label_objetivo = 'Objetivo (Final)' if i == len(dados_plot) - 1 else f"Waypoint {i+1}"
        
        plt.plot(inicio[0], inicio[1], 'go', markersize=8, label=label_inicio, zorder=3)
        plt.plot(objetivo[0], objetivo[1], 'bo', markersize=8, label=label_objetivo, zorder=3)
            
    # ==========================================
    # Traça as rotas vermelhas otimizadas absolutas no topo de tudo (zorder=4)
    # ==========================================
    for i, (_, caminho, _, _) in enumerate(dados_plot):
        if caminho:
            x_caminho = [p[0] for p in caminho]
            y_caminho = [p[1] for p in caminho]
            plt.plot(x_caminho, y_caminho, color='red', linewidth=2, label='Caminho Otimizado' if i == 0 else "", zorder=4)
        
    plt.legend()
    print("Gráfico completo gerado! Você pode salvá-lo. Feche a janela para iniciar a simulação no CoppeliaSim.")
    plt.show() # Mantém a janela aberta até você fechar no 'X'

def suavizar_caminho(caminho, obstaculos):
    # Se o caminho for muito curto, não precisa suavizar
    if caminho is None or len(caminho) <= 2:
        return caminho
        
    caminho_otimizado = [caminho[0]]
    indice_atual = 0
    
    while indice_atual < len(caminho) - 1:
        # Varre a lista de trás para frente procurando o ponto mais distante
        # que podemos acessar em uma linha reta perfeita sem bater em nada.
        for i in range(len(caminho) - 1, indice_atual, -1):
            if not colide_segmento(caminho[indice_atual], caminho[i], obstaculos):
                caminho_otimizado.append(caminho[i])
                indice_atual = i
                break
                
    return caminho_otimizado

# ==========================================
# 4. EXECUÇÃO PRINCIPAL (MÚLTIPLOS WAYPOINTS)
# ==========================================

# 1. Pega a lista de destinos (Ex: Ponto A, Ponto B, etc)
destinos = input_de_ponto() 
obstaculos = obstaculos_cena(nomes)

posicao_inicial_3d = posicao(barco)
ponto_atual = (posicao_inicial_3d[0], posicao_inicial_3d[1])

caminho_completo = [] # Esta lista vai guardar a rota inteira costurada
dados_plot = []       # Esta lista vai guardar as árvores e trechos para plotar de uma vez

print('Começando a planejar a rota por múltiplos waypoints...')

# 2. Loop de Planejamento: Calcula o RRT trecho por trecho
for objetivo in destinos:
    print(f"\nPlanejando trecho de {ponto_atual} para {objetivo}...")
    caminho_trecho, arvore, historico_melhores_caminhos = RRT_Star(ponto_atual, objetivo, obstaculos)  
    
    if caminho_trecho:
        print(f"\nHistórico de rotas mais rápidas encontradas durante as iterações para este trecho:")
        for reg in historico_melhores_caminhos:
            print(f"  Iteração {reg['iteracao']} | Custo {reg['custo']:.2f} | Pontos (x, y): {reg['caminho']}")
        # =========================================================
        # NOVO: Estica o barbante! Corta todos os zigue-zagues.
        caminho_trecho = suavizar_caminho(caminho_trecho, obstaculos)
        # =========================================================
        
        # Guarda os dados para plotar no final, não desenha ainda.
        dados_plot.append((arvore, caminho_trecho, ponto_atual, objetivo))
        
        # Junta o trecho novo na rota completa. 
        # (Ignoramos o índice [0] a partir do 2º trecho para não duplicar o ponto de emenda)
        if not caminho_completo:
            caminho_completo.extend(caminho_trecho)
        else:
            caminho_completo.extend(caminho_trecho[1:])
            
        # O barco virtual "chegou" no objetivo, então ele vira o ponto de partida do próximo trecho
        ponto_atual = objetivo
    else:
        print(f"Caminho bloqueado para {objetivo}. Abortando missão.")
        caminho_completo = None
        break

# Visualiza tudo de uma vez antes de iniciar a simulação
if dados_plot:
    visualizar_rrt_completo(dados_plot, obstaculos)

# 3. Execução da Simulação (Se a rota inteira foi gerada com sucesso)
if caminho_completo:
    print("\nIniciando simulação da trajetória completa no CoppeliaSim!")
    client.setStepping(True)
    dt = 0.05
    sim.setFloatParam(sim.floatparam_simulation_time_step, dt)
    sim.startSimulation() 
    
    estado_controle = 1.0
    velocidade = 1.0 # m/s
    
    x_atual, y_atual = caminho_completo[0]
    ori = sim.getObjectOrientation(barco, sim.handle_world)

    for i in range(len(caminho_completo) - 1):
        x_fim, y_fim = caminho_completo[i+1]
        print(f"Navegando para o waypoint: ({x_fim:.2f}, {y_fim:.2f})")
        
        while True:
            dist = m.hypot(x_fim - x_atual, y_fim - y_atual)
            if dist < 0.1: # Chegou no waypoint
                break
                
            # 2. Atualização do Ângulo de Referência
            i_ang = m.atan2(y_fim - y_atual, x_fim - x_atual)
            
            # 3. Chama o módulo de controle para calcular o Gain Scheduling
            s_ang = controlador.atualizar_angulo(estado_controle, i_ang)
            
            # 4. Cinemática: Atualiza a posição baseada na velocidade e ângulo
            x_atual += velocidade * m.cos(s_ang) * dt
            y_atual += velocidade * m.sin(s_ang) * dt
            
            sim.setObjectPosition(barco, sim.handle_world, [x_atual, y_atual, posicao_inicial_3d[2]])
            sim.setObjectOrientation(barco, sim.handle_world, [ori[0], ori[1], s_ang + m.pi])
            client.step()

    print("Destino final alcançado!")
    sim.stopSimulation()
