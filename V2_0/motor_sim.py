"""
motor_sim.py — Camada 3: HAL / Planta Simulada do Motor DC
============================================================
Projeto IARA — Simulação Software-in-the-Loop (SITL)

Modela a dinâmica elétrica real de um motor DC de ímã permanente:
  - Tensão aplicada com PWM
  - Inércia mecânica (filtro passa-baixa de 1ª ordem)
  - Força Contra-Eletromotriz (FCEM / Back-EMF)
  - Corrente de armadura (com pico de partida)
  - Conversão RPM → rad/s → velocidade linear da hélice

Referência: Modelo eletromecânico padrão (Ogata, 5ª ed., Cap. 4)
"""

import math


class SimulatedMotor:
    """
    Simula a planta elétrica de UM motor DC com Ponte H virtual.

    Parâmetros do construtor mapeiam um motor RS-775 típico de barco
    em escala reduzida. Todos podem ser sobrescritos pelo caller.
    """

    def __init__(
        self,
        nome: str = "Motor",
        # ── Parâmetros Elétricos ──
        tensao_fonte: float = 12.0,       # [V]  Tensão da bateria
        resistencia_armadura: float = 1.2, # [Ω]  Resistência da armadura (Ra)
        ke: float = 0.01,                 # [V/RPM] Constante de FCEM
        # ── Parâmetros Mecânicos ──
        rpm_max: float = 6000.0,          # [RPM] Limite superior do motor
        alpha_inercia: float = 0.08,      # [adim] Constante do filtro passa-baixa
        raio_helice: float | None = None, # [m]   Raio da hélice (65mm / 2)
        escala: float = 0.1,              # Escala linear da bancada
    ):
        self.nome = nome

        # Parâmetros (podem ser alterados pelo dashboard em runtime)
        self.tensao_fonte = tensao_fonte
        self.Ra = resistencia_armadura
        self.Ke = ke
        self.rpm_max = rpm_max
        self.alpha = alpha_inercia
        self.escala = escala
        self.raio_helice = 0.0325 * escala if raio_helice is None else raio_helice
        self.deadzone_pwm = 0.5 * escala

        # ── Estado Interno ──
        self.pwm = 0.0            # [%]  0–100
        self.tensao_aplicada = 0.0  # [V]
        self.rpm = 0.0            # [RPM] (valor instantâneo com inércia)
        self.fcem = 0.0           # [V]  Força Contra-Eletromotriz
        self.corrente = 0.0       # [A]  Corrente de armadura
        self.v_linear = 0.0       # [m/s] Velocidade linear na ponta da hélice
        self.rad_s = 0.0          # [rad/s] Para o CoppeliaSim

    # ──────────────────────────────────────────────
    #  Atualização a cada step de simulação
    # ──────────────────────────────────────────────
    def atualizar(self, pwm_comando: float) -> None:
        """
        Executa um ciclo completo da planta simulada.

        Parâmetros
        ----------
        pwm_comando : float
            Duty-cycle comandado em porcentagem (0–100).
            Valores negativos indicam sentido reverso.
        """
        # 0. Clamp do PWM
        self.pwm = max(-100.0, min(100.0, pwm_comando))

        # 1. Tensão Aplicada = Tensão_Fonte × (PWM / 100)
        self.tensao_aplicada = self.tensao_fonte * (self.pwm / 100.0)

        # 2. RPM-alvo instantâneo (sem inércia) — proporcional à tensão
        #    Quando a FCEM equilibra a tensão, corrente → 0 e RPM estabiliza.
        if abs(self.pwm) > self.deadzone_pwm:
            rpm_alvo = (self.pwm / 100.0) * self.rpm_max
        else:
            rpm_alvo = 0.0

        # 3. Inércia mecânica: filtro passa-baixa de 1ª ordem
        #    RPM(k) = RPM(k-1) + α × (RPM_alvo − RPM(k-1))
        erro_rpm = rpm_alvo - self.rpm
        self.rpm += self.alpha * erro_rpm

        # 4. Força Contra-Eletromotriz (FCEM = Ke × RPM)
        self.fcem = abs(self.rpm) * self.Ke

        # 5. Corrente de Armadura: I = (V_aplicada − FCEM) / Ra
        #    Na partida: RPM ≈ 0, FCEM ≈ 0, logo I = V/Ra (pico)
        #    Em regime:  FCEM → V, logo I → 0
        if abs(self.pwm) > self.deadzone_pwm:
            corrente_bruta = (abs(self.tensao_aplicada) - self.fcem) / self.Ra
            self.corrente = max(0.0, corrente_bruta)
        else:
            self.corrente = 0.0

        # 6. Conversão RPM → rad/s
        self.rad_s = self.rpm * 2.0 * math.pi / 60.0

        # 7. Velocidade linear na ponta da hélice
        #    V_linear = ω × R_hélice
        self.v_linear = self.rad_s * self.raio_helice

    # ──────────────────────────────────────────────
    #  Utilidades
    # ──────────────────────────────────────────────
    def get_estado(self) -> dict:
        """Retorna um snapshot completo do motor para log/dashboard."""
        return {
            "nome": self.nome,
            "pwm": self.pwm,
            "tensao_aplicada": self.tensao_aplicada,
            "rpm": self.rpm,
            "fcem": self.fcem,
            "corrente": self.corrente,
            "v_linear": self.v_linear,
            "rad_s": self.rad_s,
        }

    def resetar(self) -> None:
        """Zera todos os estados internos (equivalente a desligar o motor)."""
        self.pwm = 0.0
        self.tensao_aplicada = 0.0
        self.rpm = 0.0
        self.fcem = 0.0
        self.corrente = 0.0
        self.v_linear = 0.0
        self.rad_s = 0.0

    def __repr__(self) -> str:
        return (
            f"[{self.nome}] PWM={self.pwm:+6.1f}%  "
            f"V={self.tensao_aplicada:+6.2f}V  "
            f"RPM={self.rpm:+7.1f}  "
            f"I={self.corrente:5.2f}A  "
            f"FCEM={self.fcem:5.2f}V  "
            f"w={self.rad_s:+7.2f} rad/s"
        )


# ══════════════════════════════════════════════════
#  Cinemática Diferencial (Robô Diferencial)
# ══════════════════════════════════════════════════

class CinematicaDiferencial:
    """
    Converte (V, Ω) do Gain Scheduler → (V_left, V_right) para os dois motores.

    Equações:
        V_left  = V − (Ω × L / 2)
        V_right = V + (Ω × L / 2)
    """

    def __init__(self, L: float = 0.132):
        """
        Parâmetros
        ----------
        L : float
            Distância entre os dois propulsores [m].
            Padrão: 0.132 m (13.2 cm, conforme spec do barco IARA).
        """
        self.L = L
        self.half_L = L / 2.0

    def inversa(self, V: float, omega: float) -> tuple:
        """
        Cinemática Inversa: (V, Ω) → (V_left, V_right)

        Retorna
        -------
        (v_left, v_right) : tuple[float, float]
            Velocidades lineares desejadas para cada motor [m/s].
        """
        v_left = V - (omega * self.half_L)
        v_right = V + (omega * self.half_L)
        return v_left, v_right

    def direta(self, v_left: float, v_right: float) -> tuple:
        """
        Cinemática Direta: (V_left, V_right) → (V, Ω)

        Retorna
        -------
        (V, omega) : tuple[float, float]
        """
        V = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.L
        return V, omega

    def velocidade_para_pwm(
        self,
        v_desejada: float,
        v_max: float,
        pwm_max: float = 100.0
    ) -> float:
        """
        Mapeia uma velocidade linear desejada [m/s] para PWM [0–100%].

        Mapeamento linear simples:
            PWM = (V_desejada / V_max) × 100
        """
        if v_max <= 0:
            return 0.0
        pwm = (v_desejada / v_max) * pwm_max
        return max(-pwm_max, min(pwm_max, pwm))


# ══════════════════════════════════════════════════
#  Teste unitário rápido
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  TESTE UNITÁRIO — SimulatedMotor + CinematicaDiferencial")
    print("=" * 60)

    motor_e = SimulatedMotor(nome="Motor_E")
    motor_d = SimulatedMotor(nome="Motor_D")
    cine = CinematicaDiferencial(L=0.132)

    # Simula V = 0.3 m/s, Ω = 0.5 rad/s
    V_cmd, omega_cmd = 0.3, 0.5
    v_left, v_right = cine.inversa(V_cmd, omega_cmd)
    print(f"\nComando: V={V_cmd} m/s, w={omega_cmd} rad/s")
    print(f"Cinemática: V_left={v_left:.4f} m/s, V_right={v_right:.4f} m/s")

    # Calcula o V_max do motor para mapear para PWM
    v_max = motor_e.rpm_max * 2 * math.pi / 60.0 * motor_e.raio_helice
    pwm_e = cine.velocidade_para_pwm(v_left, v_max)
    pwm_d = cine.velocidade_para_pwm(v_right, v_max)
    print(f"PWMs: E={pwm_e:.1f}%, D={pwm_d:.1f}%")

    # Simula 50 iterações para ver inércia + pico de corrente
    print(f"\n{'Step':>4} | {motor_e.nome:^50} | {motor_d.nome:^50}")
    print("-" * 110)
    for i in range(50):
        motor_e.atualizar(pwm_e)
        motor_d.atualizar(pwm_d)
        if i < 10 or i % 5 == 0:
            print(f"{i:4d} | {motor_e!r} | {motor_d!r}")

    print("\n[OK] Teste concluido com sucesso.")
