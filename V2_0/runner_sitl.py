"""
runner_sitl.py — Runner Principal com Arquitetura SITL (4 Camadas)
==================================================================
Projeto IARA — Software-in-the-Loop

Pipeline:
    Camada 1 (Planejamento) : RRT*            → waypoints (x, y)
    Camada 2 (Controle)     : Gain Scheduler   → V, Ω
    Camada 3 (HAL)          : SimulatedMotor   → RPM, rad/s
    Camada 4 (Simulação)    : CoppeliaSim      → setJointTargetVelocity

Este arquivo substitui o runner.py original. O RRT* permanece intacto;
a diferença está na seção de execução da simulação (seção 3), que agora
passa pelos motores simulados antes de enviar comandos ao CoppeliaSim.

Uso:
    python runner_sitl.py
    python runner_sitl.py --escala-real
    python runner_sitl.py --escala=0.1
"""

import math as m
import random
import sys
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# ── Módulos SITL do Projeto IARA ──
from motor_sim import SimulatedMotor, CinematicaDiferencial
from controle import GainScheduler, InterfaceCoppelia, SITLOrchestrator
from dashboard import DashboardBancada
from escala_config import EscalaFisica


# ══════════════════════════════════════════════════
#  Variáveis Globais para o RRT*
# ══════════════════════════════════════════════════

# Limites da rota original; configurar_escala converte para a cena.
X_MIN_ORIGINAL = -0.5
X_MAX_ORIGINAL = 40.5
Y_MIN_ORIGINAL = -8.0
Y_MAX_ORIGINAL = 12.5
PONTO_B_ORIGINAL = (30.0, -7.0)
FINAL_ROTA_X_ORIGINAL = 40.0

X_MIN = X_MIN_ORIGINAL * 0.1
X_MAX = X_MAX_ORIGINAL * 0.1
Y_MIN = Y_MIN_ORIGINAL * 0.1
Y_MAX = Y_MAX_ORIGINAL * 0.1

PASSO = 0.4 * 0.1
MAX_ITERACOES = 5000
GOAL_BIAS = 0.10
RAIO_REWIRE = 1.0 * 0.1
RESOLUCAO_COLISAO = 0.03 * 0.1

# Correção fixa da orientação do modelo visual no CoppeliaSim, em radianos.
# Estes valores preservam o comportamento anterior. Altere somente se o
# modelo IARINHA precisar de uma correção de montagem.
CORRECAO_ORIENTACAO_IARINHA = (0.0, 0.0, 0.0)


# ══════════════════════════════════════════════════
#  Path Planning — RRT* (Idêntico ao original)
# ══════════════════════════════════════════════════

def distancia(a, b):
    return m.hypot(a[0] - b[0], a[1] - b[1])


def configurar_escala(escala):
    global X_MIN, X_MAX, Y_MIN, Y_MAX, PASSO, RAIO_REWIRE, RESOLUCAO_COLISAO
    X_MIN = X_MIN_ORIGINAL * escala
    X_MAX = X_MAX_ORIGINAL * escala
    Y_MIN = Y_MIN_ORIGINAL * escala
    Y_MAX = Y_MAX_ORIGINAL * escala
    PASSO = 0.4 * escala
    RAIO_REWIRE = 1.0 * escala
    RESOLUCAO_COLISAO = 0.03 * escala


def obter_escala_argumento(argumentos):
    if "--escala-real" in argumentos:
        return 1.0
    for argumento in argumentos:
        if argumento.startswith("--escala="):
            fator = float(argumento.split("=", 1)[1])
            if fator <= 0:
                raise ValueError("A escala deve ser maior que zero.")
            return fator
    return 0.1


def input_de_ponto(escala):
    texto = input(
        "Insira os waypoints em metros da proposta original, separados por ';' "
        "(Ex: 2.0,1.0 ; 3.0,-0.7): "
    )
    lista_str = texto.split(';')
    pontos = []
    for p in lista_str:
        coords = p.split(',')
        pontos.append((
            float(coords[0].strip()) * escala,
            float(coords[1].strip()) * escala,
        ))
    return pontos


def adicionar_continuacao_pos_b(destinos, escala):
    """Adiciona uma janela finita para representar B -> +infinito."""
    if not destinos:
        return destinos

    ultimo_x, ultimo_y = destinos[-1]
    ponto_b = (
        PONTO_B_ORIGINAL[0] * escala,
        PONTO_B_ORIGINAL[1] * escala,
    )
    tolerancia = 1e-6
    chegou_em_b = (
        abs(ultimo_x - ponto_b[0]) <= tolerancia
        and abs(ultimo_y - ponto_b[1]) <= tolerancia
    )
    if chegou_em_b:
        destinos = list(destinos)
        destinos.append((
            FINAL_ROTA_X_ORIGINAL * escala,
            PONTO_B_ORIGINAL[1] * escala,
        ))
    return destinos


def limites_bbox_mundo(sim, handle):
    """Retorna os limites XY da caixa delimitadora do objeto no mundo."""
    min_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
    max_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x)
    min_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
    max_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y)
    min_z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
    max_z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z)

    matriz = sim.getObjectMatrix(handle, sim.handle_world)
    pontos_mundo = []
    for x in (min_x, max_x):
        for y in (min_y, max_y):
            for z in (min_z, max_z):
                pontos_mundo.append((
                    matriz[0] * x + matriz[1] * y + matriz[2] * z + matriz[3],
                    matriz[4] * x + matriz[5] * y + matriz[6] * z + matriz[7],
                ))

    xs = [ponto[0] for ponto in pontos_mundo]
    ys = [ponto[1] for ponto in pontos_mundo]
    return min(xs), max(xs), min(ys), max(ys)


def obstaculos_cena(sim, nome_obstaculos, margem=2.333):
    """Carrega obstáculos do CoppeliaSim. Se não conectado, retorna lista vazia."""
    lista_obstaculos = []
    if sim is None:
        return lista_obstaculos

    for nome in nome_obstaculos:
        try:
            handle = sim.getObject(f'/{nome}')
            pos = sim.getObjectPosition(handle, sim.handle_world)

            min_x, max_x, min_y, max_y = limites_bbox_mundo(sim, handle)
            centro_x = (min_x + max_x) / 2.0
            centro_y = (min_y + max_y) / 2.0
            largura = (max_x - min_x) + margem
            profundidade = (max_y - min_y) + margem

            lista_obstaculos.append((centro_x, centro_y, largura, profundidade))
            print(f"Obstáculo '{nome}' carregado: X={centro_x:.2f}, Y={centro_y:.2f}, "
                  f"Larg={largura:.2f}, Prof={profundidade:.2f}")

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
        x_min = ox - (larg / 2.0)
        x_max = ox + (larg / 2.0)
        y_min = oy - (prof / 2.0)
        y_max = oy + (prof / 2.0)
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return False
    return True


def colide_segmento(a, b, obstaculos, resolucao=None):
    if resolucao is None:
        resolucao = RESOLUCAO_COLISAO
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
    if d == 0:
        return origem
    if d <= passo:
        return destino
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
        return None, None, []

    arvore = [{"ponto": posicao_inicial, "pai": None, "custo": 0.0}]
    indice_objetivo = None
    melhor_custo = float('inf')
    historico_melhores_caminhos = []

    for iteracao in range(MAX_ITERACOES):
        q_rand = objetivo if random.random() < GOAL_BIAS else ponto_aleatorio()

        pontos_arvore = [no["ponto"] for no in arvore]
        kdtree = KDTree(pontos_arvore)

        _, indice_near = kdtree.query(q_rand)
        q_near = arvore[indice_near]["ponto"]
        q_new = steer(q_near, q_rand, PASSO)

        if distancia(q_near, q_new) < 1e-5:
            continue

        if not ponto_valido(q_new, obstaculos) or colide_segmento(q_near, q_new, obstaculos):
            continue

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

        for idx_viz in indices_vizinhos:
            if idx_viz == melhor_pai:
                continue
            custo_via_novo = melhor_custo_novo + distancia(q_new, arvore[idx_viz]["ponto"])
            if custo_via_novo < arvore[idx_viz]["custo"]:
                if not colide_segmento(q_new, arvore[idx_viz]["ponto"], obstaculos):
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
                        _propagar_custo(arvore, idx_viz)

        if distancia(q_new, objetivo) <= PASSO and not colide_segmento(q_new, objetivo, obstaculos):
            custo_chegada = melhor_custo_novo + distancia(q_new, objetivo)
            if custo_chegada < melhor_custo:
                if indice_objetivo is not None:
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

    if indice_objetivo is not None:
        print(f"RRT* finalizado! Melhor caminho com custo={melhor_custo:.2f} e {len(arvore)} nós na árvore.")
        return reconstruir_caminho(arvore, indice_objetivo), arvore, historico_melhores_caminhos
    else:
        print("RRT* não encontrou caminho.")
        return None, arvore, []


def visualizar_rrt_completo(dados_plot, obstaculos):
    plt.figure(figsize=(10, 10))
    import matplotlib.patches as patches

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

    for arvore, _, _, _ in dados_plot:
        for no in arvore:
            if no["pai"] is not None:
                ponto_atual = no["ponto"]
                ponto_pai = arvore[no["pai"]]["ponto"]
                plt.plot([ponto_pai[0], ponto_atual[0]], [ponto_pai[1], ponto_atual[1]],
                         color='cyan', alpha=0.3, linewidth=1, zorder=2)

    for i, (_, _, inicio, objetivo) in enumerate(dados_plot):
        label_inicio = 'Início' if i == 0 else ""
        label_objetivo = 'Objetivo (Final)' if i == len(dados_plot) - 1 else f"Waypoint {i+1}"
        plt.plot(inicio[0], inicio[1], 'go', markersize=8, label=label_inicio, zorder=3)
        plt.plot(objetivo[0], objetivo[1], 'bo', markersize=8, label=label_objetivo, zorder=3)

    for i, (_, caminho, _, _) in enumerate(dados_plot):
        if caminho:
            x_caminho = [p[0] for p in caminho]
            y_caminho = [p[1] for p in caminho]
            plt.plot(x_caminho, y_caminho, color='red', linewidth=2,
                     label='Caminho Otimizado' if i == 0 else "", zorder=4)

    plt.legend()
    print("Gráfico completo gerado! Feche a janela para iniciar a simulação SITL.")
    plt.show()


def suavizar_caminho(caminho, obstaculos):
    if caminho is None or len(caminho) <= 2:
        return caminho

    caminho_otimizado = [caminho[0]]
    indice_atual = 0

    while indice_atual < len(caminho) - 1:
        for i in range(len(caminho) - 1, indice_atual, -1):
            if not colide_segmento(caminho[indice_atual], caminho[i], obstaculos):
                caminho_otimizado.append(caminho[i])
                indice_atual = i
                break

    return caminho_otimizado


# ══════════════════════════════════════════════════
#  EXECUÇÃO PRINCIPAL
# ══════════════════════════════════════════════════

def main():
    mostrar_dashboard = "--sem-dashboard" not in sys.argv
    escala = obter_escala_argumento(sys.argv[1:])
    escala_fisica = EscalaFisica(escala)
    configurar_escala(escala)
    nomes_obstaculos = ['Cuboid_1', 'Cuboid_2', 'Cuboid_3']

    print("=" * 60)
    print("  PROJETO IARA — Runner SITL (4 Camadas)")
    print("=" * 60)
    print(f"  Escala linear: 1:{1 / escala:g}")

    # ── Inicializar Camada 4: CoppeliaSim ──
    coppelia = InterfaceCoppelia()
    coppelia_ok = coppelia.conectar(
        nome_barco="/IARINHA",
        nome_motor_esq="/left_motor",
        nome_motor_dir="/right_motor"
    )

    sim = coppelia.sim if coppelia_ok else None

    # ── Carregar obstáculos ──
    obstaculos = obstaculos_cena(
        sim,
        nomes_obstaculos,
        margem=escala_fisica.comprimento(1.0),
    )

    # ── Posição inicial do barco ──
    if coppelia_ok:
        pos_3d = coppelia.get_posicao_barco()
        ponto_atual = (pos_3d[0], pos_3d[1])
    else:
        ponto_atual = (0.0, 0.0)
        print("Modo offline: posição inicial = (0, 0)")

    # ── Camada 1: Planejamento (RRT*) ──
    destinos = adicionar_continuacao_pos_b(input_de_ponto(escala), escala)

    caminho_completo = []
    dados_plot = []

    print('\nComeçando a planejar a rota por múltiplos waypoints...')

    for objetivo in destinos:
        print(f"\nPlanejando trecho de {ponto_atual} para {objetivo}...")
        caminho_trecho, arvore, historico = RRT_Star(ponto_atual, objetivo, obstaculos)

        if caminho_trecho:
            for reg in historico:
                print(f"  Iteração {reg['iteracao']} | Custo {reg['custo']:.2f}")

            caminho_trecho = suavizar_caminho(caminho_trecho, obstaculos)
            dados_plot.append((arvore, caminho_trecho, ponto_atual, objetivo))

            if not caminho_completo:
                caminho_completo.extend(caminho_trecho)
            else:
                caminho_completo.extend(caminho_trecho[1:])

            ponto_atual = objetivo
        else:
            print(f"Caminho bloqueado para {objetivo}. Abortando missão.")
            caminho_completo = None
            break

    # Visualizar árvore RRT*
    if dados_plot:
        visualizar_rrt_completo(dados_plot, obstaculos)

    # ── Camada 2+3+4: Execução SITL ──
    if caminho_completo:
        print("\n" + "=" * 60)
        print("  Iniciando Simulação SITL!")
        print("=" * 60)
        
        # ── Camada 2+3+4: Execução SITL ──
    if caminho_completo:
        print("\n" + "=" * 60)
        print("  Iniciando Simulação SITL!")
        print("=" * 60)

        # ---> NOVO: Desenhar waypoints no CoppeliaSim <---
        if coppelia_ok:
            # Passa a mesma tolerância usada no controle de navegação
            tolerancia_visual = escala_fisica.comprimento(1)
            coppelia.desenhar_waypoints(caminho_completo, raio_tolerancia=tolerancia_visual)

        # Criar componentes SITL
        motor_esq = SimulatedMotor(nome="Motor_E", escala=escala)
        motor_dir = SimulatedMotor(nome="Motor_D", escala=escala)
        cinematica = CinematicaDiferencial(L=escala_fisica.comprimento(0.132))
        gain_sched = GainScheduler(
            velocidade_cruzeiro=1.0,
            distancia_frenagem=escala_fisica.comprimento(0.3),
            escala=escala_fisica,
        )

        orch = SITLOrchestrator(
            coppelia=coppelia,
            gain_scheduler=gain_sched,
            motor_esq=motor_esq,
            motor_dir=motor_dir,
            cinematica=cinematica,
        )

        dt = 0.05
        # A configuração ocorre antes de iniciar a simulação física.
        dash = DashboardBancada(orch)
        if not dash.configurar():
            print("Execução cancelada na configuração.")
            return

        if coppelia_ok:
            coppelia.iniciar_simulacao(dt=dt)

        navegacao_concluida = False
        try:
            dash.rodar_navegacao(
                caminho_completo,
                dt=dt,
                configuracao_confirmada=True,
                mostrar_dashboard=mostrar_dashboard,
                tolerancia=escala_fisica.comprimento(0.05),
                correcao_orientacao=CORRECAO_ORIENTACAO_IARINHA,
            )
            navegacao_concluida = True
        except RuntimeError as erro:
            print(f"Navegação interrompida: {erro}")
        finally:
            if coppelia_ok:
                coppelia.apagar_waypoints()
                coppelia.parar_simulacao()

        if navegacao_concluida:
            print("Destino final alcançado!")
    else:
        print("Nenhum caminho foi gerado. Encerrando.")


if __name__ == "__main__":
    main()
