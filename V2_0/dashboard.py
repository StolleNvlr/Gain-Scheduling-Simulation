"""
dashboard.py — Camada Visual: Bancada Virtual / Osciloscópio OpenCV
====================================================================
Projeto IARA — Simulação Software-in-the-Loop (SITL)

Interface de teste que substitui o hardware real:
  - Sliders para forçar Tensão da Fonte e PWM (malha aberta)
  - Chave para alternar entre malha aberta e malha fechada
  - Gráficos de bancada plotados diretamente no frame OpenCV:
      • RPM (esquerdo + direito)
      • Corrente (mostra pico de partida)
      • Tensão aplicada
      • V_linear (velocidade na hélice)

Não usa matplotlib para não travar o loop — tudo renderizado via cv2.
"""

import cv2
import numpy as np
import math
import time
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from motor_sim import SimulatedMotor, CinematicaDiferencial
from controle import GainScheduler, InterfaceCoppelia, SITLOrchestrator


# ══════════════════════════════════════════════════
#  Constantes Visuais
# ══════════════════════════════════════════════════

LARGURA = 1280
ALTURA = 820
BG_COLOR = (20, 20, 30)           # Fundo escuro
GRID_COLOR = (40, 40, 55)         # Grade
TEXT_COLOR = (200, 210, 220)      # Texto principal
TITLE_COLOR = (100, 200, 255)     # Títulos dos gráficos
ACCENT_1 = (0, 200, 100)         # Verde — Motor Esquerdo
ACCENT_2 = (100, 100, 255)       # Azul — Motor Direito
WARNING_COLOR = (0, 100, 255)    # Laranja (BGR) — Alerta de corrente
SLIDER_AREA_COLOR = (30, 30, 45) # Fundo dos sliders

GRAPH_SAMPLES = 100              # Número de amostras no gráfico
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN


# ══════════════════════════════════════════════════
#  Funções auxiliares de desenho
# ══════════════════════════════════════════════════

def desenhar_grafico(
    frame: np.ndarray,
    x0: int, y0: int, w: int, h: int,
    dados_1: list, dados_2: list,
    titulo: str,
    unidade: str,
    cor_1: tuple = ACCENT_1,
    cor_2: tuple = ACCENT_2,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    label_1: str = "Esq",
    label_2: str = "Dir",
):
    """
    Desenha um gráfico de linha dupla (motor E e D) diretamente no frame.
    Sem matplotlib — usa apenas cv2.line e cv2.putText.
    """
    # ── Fundo do gráfico ──
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (25, 25, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (60, 60, 80), 1)

    # ── Título ──
    cv2.putText(frame, f"{titulo} ({unidade})", (x0 + 8, y0 + 18),
                FONT, 0.5, TITLE_COLOR, 1, cv2.LINE_AA)

    # Região de plotagem (com margem interna)
    mx, my = 45, 28  # margens
    pw = w - mx - 10
    ph = h - my - 15

    if not dados_1 and not dados_2:
        cv2.putText(frame, "Aguardando dados...", (x0 + mx, y0 + my + ph // 2),
                    FONT_SMALL, 1.0, (80, 80, 100), 1, cv2.LINE_AA)
        return

    # ── Escala Y ──
    todos = dados_1 + dados_2
    if y_min is None:
        y_min = min(todos) if todos else 0
    if y_max is None:
        y_max = max(todos) if todos else 1
    # Evita escala zero
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1
        y_max += 1

    # ── Grade horizontal ──
    for i in range(5):
        gy = y0 + my + int(ph * i / 4)
        cv2.line(frame, (x0 + mx, gy), (x0 + mx + pw, gy), GRID_COLOR, 1)
        val = y_max - (y_max - y_min) * i / 4
        cv2.putText(frame, f"{val:6.1f}", (x0 + 2, gy + 4),
                    FONT_SMALL, 0.8, (100, 100, 120), 1, cv2.LINE_AA)

    def _plotar_serie(dados, cor):
        n = len(dados)
        if n < 2:
            return
        pontos = []
        for i, val in enumerate(dados[-GRAPH_SAMPLES:]):
            fx = x0 + mx + int(pw * i / (GRAPH_SAMPLES - 1))
            # Clamp
            val_c = max(y_min, min(y_max, val))
            fy = y0 + my + int(ph * (1.0 - (val_c - y_min) / (y_max - y_min)))
            pontos.append((fx, fy))
        for i in range(len(pontos) - 1):
            cv2.line(frame, pontos[i], pontos[i + 1], cor, 2, cv2.LINE_AA)

    _plotar_serie(dados_1, cor_1)
    _plotar_serie(dados_2, cor_2)

    # ── Legenda ──
    lx = x0 + mx + pw - 120
    ly = y0 + my + 14
    cv2.circle(frame, (lx, ly), 4, cor_1, -1)
    cv2.putText(frame, label_1, (lx + 8, ly + 4), FONT_SMALL, 0.9, cor_1, 1, cv2.LINE_AA)
    cv2.circle(frame, (lx + 55, ly), 4, cor_2, -1)
    cv2.putText(frame, label_2, (lx + 63, ly + 4), FONT_SMALL, 0.9, cor_2, 1, cv2.LINE_AA)

    # ── Valores atuais ──
    val_1 = dados_1[-1] if dados_1 else 0
    val_2 = dados_2[-1] if dados_2 else 0
    cv2.putText(frame, f"E:{val_1:7.2f} D:{val_2:7.2f}",
                (x0 + mx, y0 + h - 4), FONT_SMALL, 0.9, TEXT_COLOR, 1, cv2.LINE_AA)


def desenhar_painel_info(frame, x0, y0, motor_e, motor_d, modo_str):
    """Desenha o painel de informações numéricas no canto."""
    cv2.rectangle(frame, (x0, y0), (x0 + 250, y0 + 190), SLIDER_AREA_COLOR, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + 250, y0 + 190), (60, 60, 80), 1)
    cv2.putText(frame, "PAINEL SITL", (x0 + 8, y0 + 20),
                FONT, 0.55, TITLE_COLOR, 1, cv2.LINE_AA)

    linhas = [
        f"Modo: {modo_str}",
        f"",
        f"Motor E: {motor_e.rpm:+7.1f} RPM",
        f"  I={motor_e.corrente:5.2f}A  V={motor_e.tensao_aplicada:+6.2f}V",
        f"  w={motor_e.rad_s:+7.2f} rad/s",
        f"",
        f"Motor D: {motor_d.rpm:+7.1f} RPM",
        f"  I={motor_d.corrente:5.2f}A  V={motor_d.tensao_aplicada:+6.2f}V",
        f"  w={motor_d.rad_s:+7.2f} rad/s",
    ]
    for i, txt in enumerate(linhas):
        cv2.putText(frame, txt, (x0 + 10, y0 + 40 + i * 17),
                    FONT_SMALL, 1.0, TEXT_COLOR, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════
#  Dashboard Principal
# ══════════════════════════════════════════════════

class DashboardBancada:
    """
    Interface OpenCV que funciona como um osciloscópio + painel de testes.

    Modos:
      - Malha Aberta: Sliders controlam PWM e Tensão diretamente.
      - Malha Fechada: O Gain Scheduler calcula V e Ω automaticamente.
    """

    def __init__(self, orquestrador: SITLOrchestrator):
        self.orch = orquestrador
        self.janela = "IARA SITL - Bancada Virtual"
        self.janela_config = "IARA SITL - Configuracao"

        # Estado dos sliders
        self.slider_pwm_esq = 0
        self.slider_pwm_dir = 0
        self.slider_tensao = 120   # ×0.1 → 12.0V
        self.slider_modo = 1       # 0 = Malha Aberta, 1 = Malha Fechada

        self._configurado = False
        self.configuracao_confirmada = False
        self.arquivo_run = None
        self.caminho_run = None

    def _criar_janela(self):
        """Cria a janela principal, usada somente durante a execução."""
        cv2.namedWindow(self.janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.janela, LARGURA, ALTURA)
        self._configurado = True

    def _criar_janela_configuracao(self):
        """Cria a janela modal de parâmetros antes da simulação."""
        cv2.namedWindow(self.janela_config, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.janela_config, 640, 430)

        cv2.createTrackbar("Tensao da fonte x0.1 (V)", self.janela_config,
                           self.slider_tensao, 300, self._on_tensao)
        cv2.createTrackbar("PWM esquerdo (%)", self.janela_config,
                           self.slider_pwm_esq, 100, self._on_pwm_esq)
        cv2.createTrackbar("PWM direito (%)", self.janela_config,
                           self.slider_pwm_dir, 100, self._on_pwm_dir)
        cv2.createTrackbar("Modo (0 aberta / 1 fechada)", self.janela_config,
                           self.slider_modo, 1, self._on_modo)

    def _renderizar_configuracao(self):
        frame = np.full((430, 640, 3), BG_COLOR, dtype=np.uint8)
        cv2.putText(frame, "CONFIGURACAO DA BANCADA", (24, 42),
                    FONT, 0.8, TITLE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "A fonte e unica e alimenta os dois motores.",
                    (24, 78), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tensao: {self.slider_tensao * 0.1:.1f} V",
                    (24, 125), FONT, 0.65, TEXT_COLOR, 1, cv2.LINE_AA)
        modo = "MALHA FECHADA - Gain Scheduler" if self.slider_modo else "MALHA ABERTA - PWM manual"
        cv2.putText(frame, f"Modo: {modo}", (24, 160), FONT, 0.65,
                    (0, 255, 100) if self.slider_modo else WARNING_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, f"PWM manual: E={self.slider_pwm_esq}%  D={self.slider_pwm_dir}%",
                    (24, 195), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (24, 260), (616, 330), (0, 130, 0), -1)
        cv2.putText(frame, "CONFIRMAR E INICIAR SIMULACAO",
                    (130, 304), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Clique no botao ou pressione ENTER. ESC cancela.",
                    (24, 375), FONT_SMALL, 1.1, (150, 160, 180), 1, cv2.LINE_AA)
        return frame

    def _confirmar_por_mouse(self, evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONUP and 24 <= x <= 616 and 260 <= y <= 330:
            self.configuracao_confirmada = True

    def configurar(self) -> bool:
        """Exibe a configuração e só libera a execução após confirmação."""
        self._criar_janela_configuracao()
        cv2.setMouseCallback(self.janela_config, self._confirmar_por_mouse)
        self.configuracao_confirmada = False
        while not self.configuracao_confirmada:
            cv2.imshow(self.janela_config, self._renderizar_configuracao())
            tecla = cv2.waitKey(30) & 0xFF
            if tecla in (13, 10):
                self.configuracao_confirmada = True
            elif tecla == 27:
                cv2.destroyWindow(self.janela_config)
                return False
        cv2.destroyWindow(self.janela_config)
        return True

        self._configurado = True

    # ── Callbacks dos Sliders ──
    def _on_pwm_esq(self, val):
        self.slider_pwm_esq = val

    def _on_pwm_dir(self, val):
        self.slider_pwm_dir = val

    def _on_tensao(self, val):
        self.slider_tensao = val
        tensao_real = val * 0.1
        self.orch.motor_esq.tensao_fonte = tensao_real
        self.orch.motor_dir.tensao_fonte = tensao_real

    def _on_modo(self, val):
        self.slider_modo = val

    def _iniciar_gravacao(self, caminho):
        pasta = Path("runs")
        pasta.mkdir(exist_ok=True)
        identificador = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.caminho_run = pasta / f"run_{identificador}.jsonl"
        self.arquivo_run = self.caminho_run.open("w", encoding="utf-8")
        metadados = {
            "tipo": "metadados",
            "inicio": datetime.now().isoformat(),
            "caminho": caminho,
            "modo": "fechada" if self.slider_modo else "aberta",
            "tensao_fonte": self.slider_tensao * 0.1,
            "pwm_manual": [self.slider_pwm_esq, self.slider_pwm_dir],
        }
        self.arquivo_run.write(json.dumps(metadados, ensure_ascii=False) + "\n")
        self.arquivo_run.flush()

    def _gravar_amostra(self, orch, step_idx):
        if self.arquivo_run is None:
            return
        h = orch.historico
        amostra = {
            "tipo": "amostra",
            "step": step_idx,
            "rpm_esq": h["rpm_esq"][-1], "rpm_dir": h["rpm_dir"][-1],
            "corrente_esq": h["corrente_esq"][-1], "corrente_dir": h["corrente_dir"][-1],
            "tensao_esq": h["tensao_esq"][-1], "tensao_dir": h["tensao_dir"][-1],
            "v_linear_esq": h["v_linear_esq"][-1], "v_linear_dir": h["v_linear_dir"][-1],
            "V_cmd": h["V_cmd"][-1], "omega_cmd": h["omega_cmd"][-1],
            "dist_alvo": h["dist_alvo"][-1],
        }
        self.arquivo_run.write(json.dumps(amostra) + "\n")
        self.arquivo_run.flush()

    def _encerrar_gravacao(self):
        if self.arquivo_run is not None:
            self.arquivo_run.close()
            self.arquivo_run = None
            print(f"Run salva em: {self.caminho_run}")

    # ── Renderização ──
    def _renderizar_frame(self) -> np.ndarray:
        """Renderiza um frame completo com os 4 gráficos + painel."""
        frame = np.full((ALTURA, LARGURA, 3), BG_COLOR, dtype=np.uint8)

        h = self.orch.historico

        # ── Layout: 4 gráficos (2×2) ──
        gw, gh = 490, 180  # largura e altura de cada gráfico
        gap_x, gap_y = 20, 15
        base_x, base_y = 15, 10

        # 1. RPM
        desenhar_grafico(
            frame, base_x, base_y, gw, gh,
            h["rpm_esq"], h["rpm_dir"],
            titulo="RPM", unidade="rev/min",
            y_min=-500, y_max=7000,
        )

        # 2. Corrente (mostra o pico de partida)
        desenhar_grafico(
            frame, base_x + gw + gap_x, base_y, gw, gh,
            h["corrente_esq"], h["corrente_dir"],
            titulo="CORRENTE", unidade="A",
            cor_1=(0, 180, 255), cor_2=(200, 100, 255),
            y_min=0, y_max=12,
        )

        # 3. Tensão Aplicada
        desenhar_grafico(
            frame, base_x, base_y + gh + gap_y, gw, gh,
            h["tensao_esq"], h["tensao_dir"],
            titulo="TENSAO APLICADA", unidade="V",
            y_min=-15, y_max=15,
        )

        # 4. V_linear
        desenhar_grafico(
            frame, base_x + gw + gap_x, base_y + gh + gap_y, gw, gh,
            h["v_linear_esq"], h["v_linear_dir"],
            titulo="V LINEAR (helice)", unidade="m/s",
            y_min=-0.5, y_max=1.5,
        )

        # ── Painel de informações ──
        painel_y = base_y + 2 * (gh + gap_y) + 10
        modo_str = "MALHA FECHADA (Gain Sched.)" if self.slider_modo == 1 else "MALHA ABERTA (Manual)"
        desenhar_painel_info(
            frame, base_x, painel_y,
            self.orch.motor_esq, self.orch.motor_dir,
            modo_str
        )

        # ── Legenda global ──
        cv2.putText(
            frame,
            "IARA SITL — Bancada Virtual  |  [ESC] Sair  |  Sliders: PWM, Tensao, Modo",
            (base_x, ALTURA - 15), FONT, 0.45, (80, 90, 110), 1, cv2.LINE_AA
        )

        # ── Indicador visual do modo ──
        if self.slider_modo == 1:
            cv2.putText(frame, ">> MALHA FECHADA ATIVA <<",
                        (base_x + 280, painel_y + 30), FONT, 0.6, (0, 255, 100), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, ">> MALHA ABERTA (TESTE MANUAL) <<",
                        (base_x + 280, painel_y + 30), FONT, 0.6, WARNING_COLOR, 2, cv2.LINE_AA)

            # Exibe PWMs sendo aplicados
            cv2.putText(frame, f"PWM Esq: {self.slider_pwm_esq}%   PWM Dir: {self.slider_pwm_dir}%",
                        (base_x + 280, painel_y + 60), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            tensao_real = self.slider_tensao * 0.1
            cv2.putText(frame, f"Tensao Fonte: {tensao_real:.1f} V",
                        (base_x + 280, painel_y + 85), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        return frame

    # ── Loop Principal ──
    def rodar_malha_aberta(self, duracao_s: float = 60.0):
        """
        Roda o dashboard em modo stand-alone (malha aberta pura).
        Útil para testar os motores simulados sem navegar waypoints.

        Pressione ESC para sair.
        """
        if not self.configuracao_confirmada and not self.configurar():
            return
        if not self._configurado:
            self._criar_janela()

        t0 = time.time()
        step = 0

        while time.time() - t0 < duracao_s:
            # Atualiza tensão da fonte a cada step
            tensao_real = self.slider_tensao * 0.1
            self.orch.motor_esq.tensao_fonte = tensao_real
            self.orch.motor_dir.tensao_fonte = tensao_real

            if self.slider_modo == 0:
                # Malha aberta: aplica PWM dos sliders
                self.orch.step_manual(
                    float(self.slider_pwm_esq),
                    float(self.slider_pwm_dir),
                )
            else:
                # Malha fechada sem waypoints: apenas roda os motores a 0
                self.orch.step_manual(0.0, 0.0)

            # Renderiza
            frame = self._renderizar_frame()
            cv2.imshow(self.janela, frame)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                break

            step += 1

        cv2.destroyAllWindows()

    def rodar_navegacao(self, caminho: list, dt: float = 0.05,
                        tolerancia: float = 0.1,
                        configuracao_confirmada: bool = False,
                        mostrar_dashboard: bool = True,
                        correcao_orientacao: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Roda o dashboard DURANTE a navegação por waypoints.
        O callback_step do orquestrador chama a renderização a cada step.
        A janela de configuração continua sendo exibida antes da execução;
        mostrar_dashboard controla apenas a janela dos gráficos em tempo real.
        """
        if not configuracao_confirmada and not self.configurar():
            return
        if mostrar_dashboard and not self._configurado:
            self._criar_janela()
        self._iniciar_gravacao(caminho)

        def _callback(orch, step_idx):
            self._gravar_amostra(orch, step_idx)
            if not mostrar_dashboard:
                return
            frame = self._renderizar_frame()
            cv2.imshow(self.janela, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                raise KeyboardInterrupt("ESC pressionado — abortando navegação.")

        try:
            self.orch.navegar_waypoints(
                caminho, dt=dt, tolerancia=tolerancia,
                callback_step=_callback,
                modo_malha_aberta=(self.slider_modo == 0),
                pwm_forcado_esq=float(self.slider_pwm_esq),
                pwm_forcado_dir=float(self.slider_pwm_dir),
                correcao_orientacao=correcao_orientacao,
            )
        except KeyboardInterrupt:
            print("\n[Dashboard] Navegação interrompida pelo usuário.")
        finally:
            self._encerrar_gravacao()
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════
#  Execução Standalone (Teste da Bancada)
# ══════════════════════════════════════════════════

def reproduzir_run(caminho_arquivo, atraso_s: float = 0.05):
    """Reproduz uma execução JSONL gravada pelo dashboard."""
    registros = []
    with Path(caminho_arquivo).open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            registro = json.loads(linha)
            if registro.get("tipo") == "amostra":
                registros.append(registro)

    if not registros:
        raise ValueError("O arquivo de run não contém amostras.")

    janela = "IARA SITL - Replay"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, LARGURA, ALTURA)
    historico = {chave: [] for chave in (
        "rpm_esq", "rpm_dir", "corrente_esq", "corrente_dir",
        "tensao_esq", "tensao_dir", "v_linear_esq", "v_linear_dir"
    )}

    for registro in registros:
        for chave in historico:
            historico[chave].append(registro[chave])
        frame = np.full((ALTURA, LARGURA, 3), BG_COLOR, dtype=np.uint8)
        desenhar_grafico(frame, 15, 10, 490, 180, historico["rpm_esq"],
                         historico["rpm_dir"], "RPM", "rev/min", y_min=-500, y_max=7000)
        desenhar_grafico(frame, 525, 10, 490, 180, historico["corrente_esq"],
                         historico["corrente_dir"], "CORRENTE", "A", y_min=0, y_max=12)
        desenhar_grafico(frame, 15, 205, 490, 180, historico["tensao_esq"],
                         historico["tensao_dir"], "TENSAO APLICADA", "V", y_min=-15, y_max=15)
        desenhar_grafico(frame, 525, 205, 490, 180, historico["v_linear_esq"],
                         historico["v_linear_dir"], "V LINEAR", "m/s", y_min=-0.5, y_max=1.5)
        cv2.putText(frame, f"Replay: step {registro['step']} | ESC para sair",
                    (15, ALTURA - 20), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.imshow(janela, frame)
        if cv2.waitKey(max(1, int(atraso_s * 1000))) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--replay":
    reproduzir_run(sys.argv[2])
elif __name__ == "__main__":
    print("=" * 60)
    print("  IARA SITL — Bancada Virtual Standalone")
    print("  Use os sliders para testar PWM e Tensão")
    print("  ESC para sair")
    print("=" * 60)

    # Cria os componentes
    motor_esq = SimulatedMotor(nome="Motor_E")
    motor_dir = SimulatedMotor(nome="Motor_D")
    cinematica = CinematicaDiferencial(L=0.132)
    gain_sched = GainScheduler(velocidade_cruzeiro=0.4)

    # Tenta conectar ao CoppeliaSim (não é obrigatório para a bancada)
    coppelia = InterfaceCoppelia()
    coppelia.conectar()

    # Monta o orquestrador
    orch = SITLOrchestrator(
        coppelia=coppelia,
        gain_scheduler=gain_sched,
        motor_esq=motor_esq,
        motor_dir=motor_dir,
        cinematica=cinematica,
    )

    # Roda o dashboard
    dash = DashboardBancada(orch)
    dash.rodar_malha_aberta(duracao_s=300)  # 5 minutos max
