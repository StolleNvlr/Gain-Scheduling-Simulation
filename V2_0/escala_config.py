"""Configuracao de escala entre a proposta original e a simulacao."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalaFisica:
    """Converte grandezas lineares da escala original para a simulacao."""

    fator: float = 0.1

    def __post_init__(self):
        if self.fator <= 0:
            raise ValueError("O fator de escala deve ser maior que zero.")

    def comprimento(self, valor: float) -> float:
        return valor * self.fator

    def velocidade(self, valor: float) -> float:
        return valor * self.fator

    def area(self, valor: float) -> float:
        return valor * self.fator * self.fator

    def posicao_original(self, valor: float) -> float:
        return valor / self.fator
