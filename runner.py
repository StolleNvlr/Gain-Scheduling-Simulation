from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math as m
import controle_iara
import random
client = RemoteAPIClient()
sim = client.require('sim')
tempo_controle, resposta_controle = controle_iara.dinamica()
'''
variaveis globais para o funcionamento do RRT
'''
X_MIN = -10.0
X_MAX = 100.0
Y_MIN = -10.0
Y_MAX = 100.0

PASSO = 4.0
MAX_ITERACOES = 10000
GOAL_BIAS = 0.10

nomes = ['Planta', 'Planta_2', 'Planta_3', 'Planta_4', 'Planta_5', 'Planta_6', 'Planta_7', 'Planta_8', 'Planta_9']
barco = sim.getObject('/iara')

client.setStepping(True)
sim.setFloatParam(sim.floatparam_simulation_time_step, 1.0)
'''
Path Planning em RRT
'''
def distancia(a,b):
    return m.hypot(a[0]-b[0], a[1]-b[1])

def input_de_ponto():
    ponto = input("Insira as coordenadas do ponto (x,y): ")
    return tuple(map(float, ponto.split(',')))

def posicao(barco_handle):
    posicao_inicial = sim.getObjectPosition(barco_handle, sim.handle_world)
    return posicao_inicial
def obstaculos_cena(nome_obstaculos, margem = 0.5):
    RAIO_DA_PLANTA = 0.1
    lista_obstaculos = []
    for nome in nome_obstaculos:
        try:
            # 1. Pega o identificador (Handle) do objeto pelo nome
            # O '/' na frente garante que ele procure na raiz da cena
            handle = sim.getObject(f'/{nome}')
            
            # 2. Pega a posição X e Y no mundo
            pos = sim.getObjectPosition(handle, sim.handle_world)
            centro_x = pos[0]
            centro_y = pos[1]
            
            # 3. Pega a Bounding Box (Tamanho do objeto)
            # Retorna [minX, minY, minZ, maxX, maxY, maxZ]
            #bbox = sim.getObjectBoundingBox(handle)
            
            # Calcula a largura (X) e profundidade (Y) do objeto
            #largura = bbox[3] - bbox[0]
            #profundidade = bbox[4] - bbox[1]
            
            # Pega a maior dimensão e divide por 2 para achar o raio que cobre o objeto
           # raio_objeto = max(largura, profundidade) / 2.0
            raio_total = RAIO_DA_PLANTA + margem
            
            # Adiciona no formato que o seu algoritmo RRT espera: (x, y, raio)
            lista_obstaculos.append((centro_x, centro_y, raio_total))
            print(f"Obstáculo '{nome}' carregado: X={centro_x:.2f}, Y={centro_y:.2f}, Raio={raio_total:.2f}m")
            
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o obstáculo '{nome}'. Verifique se o nome está exato no CoppeliaSim. Erro: {e}")
            
    return lista_obstaculos

def ponto_aleatorio():
    return (random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX))

def ponto_valido(p, obstaculos):
    x, y = p
    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
        return False
    for ox, oy, raio in obstaculos:
        if distancia(p, (ox, oy)) <= raio:
            return False
    return True

def colide_segmento(a, b, obstaculos, resolucao=0.5):
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

def RRT(posicao_inicial, objetivo, obstaculos):
    if not ponto_valido(posicao_inicial, obstaculos) or not ponto_valido(objetivo, obstaculos):
        print("Erro: Ponto inicial ou objetivo estão dentro de um obstáculo ou fora dos limites!")
        return None

    arvore = [{"ponto": posicao_inicial, "pai": None}]

    for iteracao in range(MAX_ITERACOES):
        q_rand = objetivo if random.random() < GOAL_BIAS else ponto_aleatorio()
        indice_near = no_mais_proximo(arvore, q_rand)
        q_near = arvore[indice_near]["ponto"]
        q_new = steer(q_near, q_rand, PASSO)

        if ponto_valido(q_new, obstaculos) and not colide_segmento(q_near, q_new, obstaculos):
            arvore.append({"ponto": q_new, "pai": indice_near})
            indice_new = len(arvore) - 1

            if distancia(q_new, objetivo) <= PASSO and not colide_segmento(q_new, objetivo, obstaculos):
                arvore.append({"ponto": objetivo, "pai": indice_new})
                print(f"Caminho RRT encontrado com {len(arvore)} nós.")
                return reconstruir_caminho(arvore, len(arvore) - 1)

    print("RRT não encontrou caminho.")
    return None
    
objetivo = input_de_ponto()
posicao_inicial_3d = posicao(barco)
posicao_inicial = (posicao_inicial_3d[0], posicao_inicial_3d[1])
obstaculos = obstaculos_cena(nomes)
print('começando a calcular a rota via função RRT')
caminho = RRT(posicao_inicial, objetivo, obstaculos)  

if caminho:
    # 4.3 Configura o modo síncrono e Inicia a Simulação
    client.setStepping(True)
    sim.setFloatParam(sim.floatparam_simulation_time_step, 0.05)
    sim.startSimulation() 

    # 4.4 Executa a navegação ponto a ponto
    for i in range(len(caminho) - 1):
        x_inicio, y_inicio = caminho[i]
        x_fim, y_fim = caminho[i+1]
        
        # ====================================================
        # NOVO: Gira o barco para apontar para o alvo
        # ====================================================
        dx = x_fim - x_inicio
        dy = y_fim - y_inicio
        angulo_yaw = m.atan2(dy, dx) + m.pi # Calcula o ângulo da trajetória em radianos
        
        # Lê a orientação atual para não capotar o barco (mantém Roll e Pitch)
        ori = sim.getObjectOrientation(barco, sim.handle_world)
        
        # Aplica a rotação apenas no eixo Z (Yaw)
        sim.setObjectOrientation(barco, sim.handle_world, [ori[0], ori[1], angulo_yaw])
        # ====================================================
        
        print(f"Navegando para o waypoint: ({x_fim:.2f}, {y_fim:.2f})")
        
        for indice in range(len(resposta_controle)):
            fator = resposta_controle[indice]
            x_inst = x_inicio + (x_fim - x_inicio) * fator
            y_inst = y_inicio + (y_fim - y_inicio) * fator
            
            sim.setObjectPosition(barco, sim.handle_world, [x_inst, y_inst, posicao_inicial_3d[2]])
            client.step()
            
            if fator >= 0.99:
                break

    print("Destino alcançado!")
    sim.stopSimulation()