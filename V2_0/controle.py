"""
controle.py — Controle original IARA + Interface CoppeliaSim (Camada 4)
==========================================================================
Projeto IARA — Simulação Software-in-the-Loop (SITL)

Contém:
  1. GainScheduler — Aplica a equação a diferenças discretizada do
      projeto_barco.m, usando os coeficientes implementados em controle_iara.py.

  2. InterfaceCoppelia — Encapsula a conexão ZeroMQ com CoppeliaSim
     e expõe métodos para enviar rad/s para as juntas dos motores.

  3. SITLOrchestrator — Roda o loop de 4 camadas em cascata.
"""

import math
import logging
from typing import Any

from escala_config import EscalaFisica

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("controle_sitl")


# ══════════════════════════════════════════════════
#  Camada 2 — Gain Scheduler
# ══════════════════════════════════════════════════

class GainScheduler:
    """
        Adaptador do controlador original do professor para o pipeline SITL.

        A saída angular s_ang é calculada pela mesma equação a diferenças do
        projeto_barco.m. O SITL precisa adicionalmente de (V, omega), portanto
        omega apenas converte o erro entre s_ang e o rumo atual em velocidade
        angular para a cinemática diferencial.
    """

    # Coeficientes da equação a diferenças — extraídos da Transformada Z
    # Formato: {estado_vel: (a1, a2, b1, b2)}
    COEFICIENTES = {
        # c2d(G, 1) para u = un, u = 0.8*un e u = 0.1*un.
        1.0: (0.279, -0.002798, 0.9669, -0.2431),
        0.8: (0.394, -0.0006436, 0.9738, -0.3671),
        0.1: (0.9028, -1.002e-16, 0.9991, -0.9019),
    }

    def __init__(
        self,
        velocidade_cruzeiro: float = 0.4,
        distancia_frenagem: float = 0.5,
        escala: float | EscalaFisica = 0.1,
    ):
        """
        Parâmetros
        ----------
        velocidade_cruzeiro : float
            Velocidade linear desejada em cruzeiro [m/s].
        distancia_frenagem : float
            Mantido por compatibilidade com os chamadores existentes. A
            velocidade do controlador original não usa frenagem por distância.
        """
        self.v_cruzeiro = velocidade_cruzeiro
        self.dist_frenagem = distancia_frenagem
        self.escala = escala if isinstance(escala, EscalaFisica) else EscalaFisica(escala)

        # Memória dos regressores (filtro Z)
        self.i1 = 0.0  # entrada(k-1)
        self.i2 = 0.0  # entrada(k-2)
        self.s1 = 0.0  # saída(k-1)
        self.s2 = 0.0  # saída(k-2)

    def resetar(self):
        """Zera a memória do filtro (usar quando mudar de waypoint)."""
        self.i1 = self.i2 = self.s1 = self.s2 = 0.0

    def _selecionar_estado(self, v_atual: float) -> float:
        """
        Seleciona o conjunto de coeficientes mais próximo da velocidade medida.

        Mantido como fallback para chamadas que não informam o estado do
        trecho da rota.
        """
        if v_atual <= 0.15:
            return 0.1
        elif v_atual <= 0.9:
            return 0.8
        else:
            return 1.0

    def velocidade_original_por_posicao(self, x_barco: float) -> float:
        """Retorna o estado de velocidade usado na simulação do MATLAB."""
        x_original = self.escala.posicao_original(x_barco)
        if x_original < 20.0:
            return 0.8
        if x_original < 30.0:
            return 0.1
        return 1.0

    def calcular(
        self,
        x_barco: float, y_barco: float, theta_barco: float,
        x_alvo: float, y_alvo: float,
        v_atual: float = 1.0,
        dt: float = 0.05,
        estado_velocidade: float | None = None
    ) -> tuple:
        """
        Calcula (V, Ω) para o step atual.

        Parâmetros
        ----------
        x_barco, y_barco : float
            Posição atual do barco [m].
        theta_barco : float
            Ângulo atual (yaw) do barco [rad].
        x_alvo, y_alvo : float
            Posição do próximo waypoint [m].
        v_atual : float
            Velocidade linear atual do barco [m/s] (usada para gain scheduling).
        dt : float
            Período do loop do SITL [s], usado apenas para converter a
            referência angular em velocidade angular.
        estado_velocidade : float ou None
            Estado de velocidade do trecho. Quando informado, reproduz o
            escalonamento por posição do projeto_barco.m.

        Retorna
        -------
        (V, omega) : tuple[float, float]
            V em [m/s], omega em [rad/s].
        """
        # 1. Ângulo desejado (referência)
        dx = x_alvo - x_barco
        dy = y_alvo - y_barco
        i_ang = math.atan2(dy, dx)

        # 2. Distância ao alvo
        dist = math.hypot(dx, dy)

        # 4. Gain Scheduling — seleciona os coeficientes
        estado = (
            estado_velocidade
            if estado_velocidade is not None
            else self._selecionar_estado(self.escala.posicao_original(v_atual))
        )
        V_original = estado if estado_velocidade is not None else self.v_cruzeiro
        V = self.escala.velocidade(V_original)
        a1, a2, b1, b2 = self.COEFICIENTES[estado]

        # 5. Equação a diferenças do projeto_barco.m
        s_ang = a1 * self.s1 + a2 * self.s2 + b1 * self.i1 + b2 * self.i2

        # 6. Atualiza memória
        self.i2 = self.i1
        self.i1 = i_ang
        self.s2 = self.s1
        self.s1 = s_ang

        # 7. Faz a ponte da referência angular original para a cinemática.
        erro_angular = s_ang - theta_barco
        erro_angular = math.atan2(math.sin(erro_angular), math.cos(erro_angular))
        omega = erro_angular / max(dt, 1e-6)

        return V, omega, s_ang, dist


# ══════════════════════════════════════════════════
#  Camada 4 — Interface CoppeliaSim via ZeroMQ
# ══════════════════════════════════════════════════

class InterfaceCoppelia:
    """
    Conecta-se ao CoppeliaSim via ZeroMQ Remote API e envia
    velocidades angulares (rad/s) para as juntas dos motores.

    Usa a API padrão coppeliasim-zmqremoteapi-client.
    """

    def __init__(self):
        self.client: Any = None
        self.sim: Any = None
        self.barco_handle = None
        self.motor_esq_handle = None
        self.motor_dir_handle = None
        self.conectado = False

    def conectar(
        self,
        nome_barco: str = "/IARINHA",
        nome_motor_esq: str = "/left_motor",
        nome_motor_dir: str = "/right_motor"
    ) -> bool:
        """
        Tenta conectar ao CoppeliaSim e obter os handles dos objetos.
        Retorna True se conseguiu, False caso contrário.
        """
        def obter_handle(candidatos, obrigatorio=True):
            ultimo_erro: Exception | None = None
            for candidato in candidatos:
                try:
                    return self.sim.getObject(candidato)
                except Exception as erro:
                    ultimo_erro = erro
            if obrigatorio:
                if ultimo_erro is not None:
                    raise ultimo_erro
                raise RuntimeError("Nenhum alias de objeto foi informado.")
            return None

        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self.client = RemoteAPIClient()
            self.sim = self.client.require('sim')

            self.barco_handle = obter_handle([
                nome_barco, "/iara", "/boat", "/Boat"
            ])
            self.motor_esq_handle = obter_handle([
                nome_motor_esq, "/left_motor", "/Left_Propeller",
                "/iara/left_motor", "/iara/Left_Propeller"
            ], obrigatorio=False)
            self.motor_dir_handle = obter_handle([
                nome_motor_dir, "/right_motor", "/Right_Propeller",
                "/iara/right_motor", "/iara/Right_Propeller"
            ], obrigatorio=False)

            self.conectado = True
            logger.info(
                f"CoppeliaSim conectado: barco={self.barco_handle}, "
                f"motor_E={self.motor_esq_handle}, motor_D={self.motor_dir_handle}"
            )
            if self.motor_esq_handle is None or self.motor_dir_handle is None:
                logger.warning(
                    "Barco conectado, mas um ou ambos os motores não foram "
                    "encontrados; a pose e os obstáculos continuarão disponíveis."
                )
            return True

        except Exception as e:
            logger.warning(f"CoppeliaSim não disponível: {e}")
            logger.info("Rodando em modo offline (sem CoppeliaSim).")
            self.conectado = False
            return False

    def enviar_velocidades(
        self,
        rad_s_esq: float,
        rad_s_dir: float,
        v_cmd: float = 0.0,
        omega_cmd: float = 0.0,
    ) -> None:
        """
        Envia velocidades angulares para as juntas no CoppeliaSim.

        Parâmetros
        ----------
        rad_s_esq : float
            Velocidade angular do motor esquerdo [rad/s].
        rad_s_dir : float
            Velocidade angular do motor direito [rad/s].
        v_cmd : float
            Comando linear para a odometria [m/s].
        omega_cmd : float
            Comando angular para a odometria [rad/s].
        """
        if not self.conectado:
            return

        try:
            if self.motor_esq_handle is not None and self.motor_dir_handle is not None:
                self.sim.setJointTargetVelocity(self.motor_esq_handle, rad_s_esq)
                self.sim.setJointTargetVelocity(self.motor_dir_handle, rad_s_dir)
            
            # Envia os sinais opcionais para a odometria em Lua.
            self.sim.setFloatSignal('cmd_v', float(v_cmd))
            self.sim.setFloatSignal('cmd_omega', float(omega_cmd))
        except Exception as e:
            logger.error(f"Erro ao enviar velocidades: {e}")

    def get_posicao_barco(self) -> tuple:
        """Retorna (x, y, z) do barco no mundo."""
        if not self.conectado:
            return (0.0, 0.0, 0.0)
        try:
            pos = self.sim.getObjectPosition(self.barco_handle, self.sim.handle_world)
            return tuple(pos)
        except Exception as e:
            logger.error(f"Erro ao ler posição: {e}")
            return (0.0, 0.0, 0.0)

    def get_orientacao_barco(self) -> tuple:
        """Retorna (α, β, γ) euler do barco no mundo."""
        if not self.conectado:
            return (0.0, 0.0, 0.0)
        try:
            ori = self.sim.getObjectOrientation(self.barco_handle, self.sim.handle_world)
            return tuple(ori)
        except Exception as e:
            logger.error(f"Erro ao ler orientação: {e}")
            return (0.0, 0.0, 0.0)

    def set_posicao_barco(self, x: float, y: float, z: float) -> None:
        """Teleporta o barco (para modo cinemático)."""
        if not self.conectado:
            return
        try:
            self.sim.setObjectPosition(
                self.barco_handle, self.sim.handle_world, [x, y, z]
            )
        except Exception as e:
            logger.error(f"Erro ao definir posição: {e}")

    def set_orientacao_barco(self, alpha: float, beta: float, gamma: float) -> None:
        """Define a orientação (euler) do barco."""
        if not self.conectado:
            return
        try:
            self.sim.setObjectOrientation(
                self.barco_handle, self.sim.handle_world, [alpha, beta, gamma]
            )
        except Exception as e:
            logger.error(f"Erro ao definir orientação: {e}")

    def iniciar_simulacao(self, dt: float = 0.05) -> None:
        """Inicia a simulação com stepping manual."""
        if not self.conectado:
            return
        self.client.setStepping(True)
        self.sim.setFloatParam(self.sim.floatparam_simulation_time_step, dt)
        self.sim.startSimulation()
        logger.info(f"Simulação iniciada (dt={dt}s, stepping=True)")

    def step(self) -> None:
        """Avança um passo de simulação."""
        if not self.conectado:
            return
        self.client.step()

    def parar_simulacao(self) -> None:
        """Para a simulação."""
        if not self.conectado:
            return
        self.sim.stopSimulation()
        logger.info("Simulação parada.")
    def desenhar_waypoints(self, caminho: list, raio_tolerancia: float = 0.5) -> None:
        """Desenha mastros com bandeiras e círculos tracejados nos waypoints 3D."""
        if not self.conectado:
            return
            
        import math
        
        # Salvando os handles no 'self' para podermos destruí-los depois
        self.handle_mastro = self.sim.addDrawingObject(self.sim.drawing_lines, 4, 0, -1, 9999, [1, 0.5, 0]) 
        self.handle_bandeira = self.sim.addDrawingObject(self.sim.drawing_triangles, 0, 0, -1, 9999, [1, 0, 0])
        self.handle_circulo = self.sim.addDrawingObject(self.sim.drawing_lines, 2, 0, -1, 9999, [1, 1, 0]) 
        
        for wp in caminho:
            x, y = wp[0], wp[1]
            z_chao = 0.1  
            z_topo = 1.0  
            
            self.sim.addDrawingObjectItem(self.handle_mastro, [x, y, z_chao, x, y, z_topo])
            
            self.sim.addDrawingObjectItem(self.handle_bandeira, [
                x, y, z_topo,                   
                x + 0.4, y, z_topo - 0.2,       
                x, y, z_topo - 0.4              
            ])
            
            passos = 16 
            for i in range(passos):
                if i % 2 == 0:
                    ang1 = i * (2 * math.pi / passos)
                    ang2 = (i + 1) * (2 * math.pi / passos)
                    
                    x1 = x + raio_tolerancia * math.cos(ang1)
                    y1 = y + raio_tolerancia * math.sin(ang1)
                    x2 = x + raio_tolerancia * math.cos(ang2)
                    y2 = y + raio_tolerancia * math.sin(ang2)
                    
                    self.sim.addDrawingObjectItem(self.handle_circulo, [x1, y1, z_chao, x2, y2, z_chao])

    def apagar_waypoints(self) -> None:
        """Remove os desenhos visuais da cena 3D do CoppeliaSim."""
        if not self.conectado:
            return
        try:
            if hasattr(self, 'handle_mastro'):
                self.sim.removeDrawingObject(self.handle_mastro)
            if hasattr(self, 'handle_bandeira'):
                self.sim.removeDrawingObject(self.handle_bandeira)
            if hasattr(self, 'handle_circulo'):
                self.sim.removeDrawingObject(self.handle_circulo)
        except Exception as e:
            pass # Se o simulador já fechou, ignora


# ══════════════════════════════════════════════════
#  Loop Orquestrador SITL (4 Camadas em Cascata)
# ══════════════════════════════════════════════════

class SITLOrchestrator:
    """
    Roda o loop principal do SITL:
        Camada 1: RRT* → waypoints (externo, recebido como lista)
        Camada 2: GainScheduler → V, Ω
        Camada 3: SimulatedMotor → RPM, rad/s
        Camada 4: CoppeliaSim → setJointTargetVelocity
    """

    def __init__(self, coppelia: InterfaceCoppelia, gain_scheduler: GainScheduler,
                 motor_esq, motor_dir, cinematica):
        """
        Parâmetros
        ----------
        coppelia : InterfaceCoppelia
        gain_scheduler : GainScheduler
        motor_esq, motor_dir : SimulatedMotor
        cinematica : CinematicaDiferencial
        """
        self.coppelia = coppelia
        self.gs = gain_scheduler
        self.motor_esq = motor_esq
        self.motor_dir = motor_dir
        self.cine = cinematica

        # Histórico para o dashboard (últimas N amostras)
        self.historico_tamanho = 200
        self.historico = {
            "rpm_esq": [], "rpm_dir": [],
            "corrente_esq": [], "corrente_dir": [],
            "tensao_esq": [], "tensao_dir": [],
            "v_linear_esq": [], "v_linear_dir": [],
            "V_cmd": [], "omega_cmd": [],
            "dist_alvo": [],
        }

    def _registrar_historico(self, V_cmd: float, omega_cmd: float, dist: float):
        """Adiciona amostras ao histórico circular."""
        h = self.historico
        h["rpm_esq"].append(self.motor_esq.rpm)
        h["rpm_dir"].append(self.motor_dir.rpm)
        h["corrente_esq"].append(self.motor_esq.corrente)
        h["corrente_dir"].append(self.motor_dir.corrente)
        h["tensao_esq"].append(self.motor_esq.tensao_aplicada)
        h["tensao_dir"].append(self.motor_dir.tensao_aplicada)
        h["v_linear_esq"].append(self.motor_esq.v_linear)
        h["v_linear_dir"].append(self.motor_dir.v_linear)
        h["V_cmd"].append(V_cmd)
        h["omega_cmd"].append(omega_cmd)
        h["dist_alvo"].append(dist)

        # Limita o tamanho
        for key in h:
            if len(h[key]) > self.historico_tamanho:
                h[key] = h[key][-self.historico_tamanho:]

    def navegar_waypoints(
        self,
        caminho: list,
        dt: float = 0.05,
        tolerancia: float = 0.1,
        max_passos_waypoint: int = 20000,
        max_passos_sem_progresso: int = 200,
        callback_step=None,
        modo_malha_aberta: bool = False,
        pwm_forcado_esq: float = 0.0,
        pwm_forcado_dir: float = 0.0,
        correcao_orientacao: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Navega por uma lista de waypoints [(x, y), ...] usando o pipeline SITL.

        Parâmetros
        ----------
        caminho : list of (float, float)
            Waypoints gerados pelo RRT*.
        dt : float
            Timestep de simulação [s].
        tolerancia : float
            Distância mínima para considerar waypoint atingido [m].
        max_passos_waypoint : int
            Limite de iterações para cada waypoint.
        max_passos_sem_progresso : int
            Limite de iterações consecutivas sem redução relevante da distância.
        callback_step : callable or None
            Função chamada a cada step, recebe (orchestrator, step_index).
            Útil para o dashboard atualizar seus gráficos.
        modo_malha_aberta : bool
            Se True, ignora o Gain Scheduler e usa pwm_forcado.
        pwm_forcado_esq, pwm_forcado_dir : float
            PWM forçado (0–100%) para teste em malha aberta.
        correcao_orientacao : tuple[float, float, float]
            Correção fixa (alpha, beta, yaw) aplicada à orientação do modelo,
            em radianos. O yaw atual continua sendo calculado pelo controlador.
        """
        if not caminho or len(caminho) < 2:
            logger.warning("Caminho vazio ou com menos de 2 pontos.")
            return

        # Obter posição/orientação inicial
        if self.coppelia.conectado:
            pos = self.coppelia.get_posicao_barco()
            ori = self.coppelia.get_orientacao_barco()
            x_atual, y_atual, z_base = pos[0], pos[1], pos[2]
            theta_atual = ori[2]
        else:
            x_atual, y_atual = caminho[0]
            z_base = 0.0
            theta_atual = 0.0

        # Calcular V_max para mapeamento PWM
        v_max = self.motor_esq.rpm_max * 2.0 * math.pi / 60.0 * self.motor_esq.raio_helice

        step_global = 0

        for idx_wp in range(len(caminho) - 1):
            x_alvo, y_alvo = caminho[idx_wp + 1]
            logger.info(f"Waypoint {idx_wp+1}/{len(caminho)-1}: ({x_alvo:.2f}, {y_alvo:.2f})")
            self.gs.resetar()
            passos_waypoint = 0
            passos_sem_progresso = 0
            menor_distancia = float("inf")

            while True:
                dist = math.hypot(x_alvo - x_atual, y_alvo - y_atual)
                if dist < tolerancia:
                    break
                if passos_waypoint >= max_passos_waypoint:
                    raise RuntimeError(
                        f"Waypoint {idx_wp + 1} excedeu {max_passos_waypoint} passos "
                        f"sem ser alcançado (distância={dist:.4f} m)."
                    )
                if dist < menor_distancia - max(1e-6, tolerancia * 0.01):
                    menor_distancia = dist
                    passos_sem_progresso = 0
                else:
                    passos_sem_progresso += 1
                if passos_sem_progresso >= max_passos_sem_progresso:
                    raise RuntimeError(
                        f"Waypoint {idx_wp + 1} estagnado após "
                        f"{max_passos_sem_progresso} passos "
                        f"(distância={dist:.4f} m, v_esq={self.motor_esq.v_linear:.6f}, "
                        f"v_dir={self.motor_dir.v_linear:.6f})."
                    )

                if modo_malha_aberta:
                    # ── Malha Aberta: PWM forçado pelo dashboard ──
                    pwm_e = pwm_forcado_esq
                    pwm_d = pwm_forcado_dir
                    V_cmd, omega_cmd = 0.0, 0.0
                    s_ang = theta_atual
                else:
                    # ── Malha Fechada: Pipeline completo ──
                    # Camada 2: Gain Scheduler
                    v_atual_estimada = (abs(self.motor_esq.v_linear) + abs(self.motor_dir.v_linear)) / 2.0
                    estado_velocidade = self.gs.velocidade_original_por_posicao(x_atual)
                    V_cmd, omega_cmd, s_ang, dist = self.gs.calcular(
                        x_atual, y_atual, theta_atual,
                        x_alvo, y_alvo,
                        v_atual=max(v_atual_estimada, 0.1),
                        dt=dt,
                        estado_velocidade=estado_velocidade
                    )

                    # Camada 3a: Cinemática Diferencial
                    v_left, v_right = self.cine.inversa(V_cmd, omega_cmd)

                    # Camada 3b: Mapeia velocidade → PWM
                    pwm_e = self.cine.velocidade_para_pwm(v_left, v_max)
                    pwm_d = self.cine.velocidade_para_pwm(v_right, v_max)

                # Camada 3c: Planta do Motor (Física Simulada)
                self.motor_esq.atualizar(pwm_e)
                self.motor_dir.atualizar(pwm_d)

                # Camada 4: Envia rad/s para CoppeliaSim
                self.coppelia.enviar_velocidades(
                    self.motor_esq.rad_s,
                    self.motor_dir.rad_s,
                    v_cmd=V_cmd,
                    omega_cmd=omega_cmd
                )

                # Atualiza posição simulada (cinemática)
                if not modo_malha_aberta:
                    v_real = (self.motor_esq.v_linear + self.motor_dir.v_linear) / 2.0
                    x_atual += v_real * math.cos(s_ang) * dt
                    y_atual += v_real * math.sin(s_ang) * dt
                    theta_atual = s_ang

                    # Atualiza pose no CoppeliaSim
                    self.coppelia.set_posicao_barco(x_atual, y_atual, z_base)
                    correcao_alpha, correcao_beta, correcao_yaw = correcao_orientacao
                    self.coppelia.set_orientacao_barco(
                        correcao_alpha,
                        correcao_beta,
                        s_ang + math.pi + correcao_yaw,
                    )

                # Registra histórico
                self._registrar_historico(V_cmd, omega_cmd, dist)

                # Step de simulação
                self.coppelia.step()

                # Callback para dashboard
                if callback_step:
                    callback_step(self, step_global)

                step_global += 1
                passos_waypoint += 1

        logger.info("Navegação concluída!")

    def step_manual(
        self,
        pwm_esq: float,
        pwm_dir: float,
    ) -> None:
        """
        Executa um único step em malha aberta (para o dashboard).
        Não navega waypoints — apenas atualiza os motores e envia ao CoppeliaSim.
        """
        self.motor_esq.atualizar(pwm_esq)
        self.motor_dir.atualizar(pwm_dir)
        self.coppelia.enviar_velocidades(
            self.motor_esq.rad_s,
            self.motor_dir.rad_s,
            v_cmd=0.0,      # Em malha aberta pura via PWM, não há comando global de V/Omega
            omega_cmd=0.0
        )
        self._registrar_historico(0.0, 0.0, 0.0)
        self.coppelia.step()
