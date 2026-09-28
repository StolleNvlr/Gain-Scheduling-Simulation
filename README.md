# Gain-Scheduling-Simulation
Simulação no CoppeliaSim de um barco autônomo integrando planejamento espacial ótimo (RRT*) e controle adaptativo (Gain Scheduling).

## Como Utilizar o Código

1. **Prepare o CoppeliaSim:** Abra a cena com o modelo `IARA` na origem e os obstáculos configurados. **Não dê play na simulação**.
2. **Execute o Roteamento:** No terminal, rode o arquivo principal:
   ```bash
   python runner.py
   ```
3. **Insira os Waypoints:** Quando solicitado, insira os pontos em escala separados por ponto-e-vírgula (`;`). Para reproduzir o artigo, digite:
   `2.0, 1.0; 3.0, -0.7; 4.0, -0.7`
4. **Validação Visual:** Feche as janelas dos gráficos gerados pelo Matplotlib para liberar o script. O barco iniciará o trajeto autonomamente no simulador.
5. **Geração de Gráficos (Opcional):** Para gerar os gráficos da resposta ao degrau da planta (Figuras 3, 4 e 5), rode isoladamente:
   ```bash
   python controle_iara.py
   ```

---

## 1. Planejamento Espacial (`runner.py`)

### 1.1 Modelagem de Obstáculos (AABB)
O mapeamento por círculos foi substituído por **Caixas Delimitadoras (Bounding Boxes)** para evitar o "efeito túnel". O ponto $(x,y)$ é invalidado se cair dentro da área inflada da rocha:
$$x_{min} \leq x \leq x_{max} \quad \text{e} \quad y_{min} \leq y \leq y_{max}$$

### 1.2 RRT* e Otimização KD-Tree
O algoritmo foi evoluído para **RRT***. Em vez de busca linear iterativa, utiliza a estrutura `scipy.spatial.KDTree` para encontrar vizinhos em raio ótimo em tempo logarítmico $\mathcal{O}(\log N)$, permitindo a reconexão contínua (*rewiring*) para reduzir o custo do percurso.

### 1.3 Suavização de Rota (Path Smoothing)
O caminho final bruto do RRT* passa por um filtro de *Line of Sight* reverso. O algoritmo traça retas projetadas ignorando nós intermediários redundantes caso não haja intersecção com os obstáculos:
$$P(t) = A + t(B - A)$$
*(Validação feita discretamente a uma alta resolução de `0.03m` por passo).*

---

## 2. Controle Adaptativo (`controle_iara.py`)

### 2.1 Gain Scheduling
A planta PD é adaptada dinamicamente conforme a velocidade exigida pelo trecho ($1.0 u_n$, $0.8 u_n$, $0.1 u_n$). Para manter um amortecimento constante sem sobressinal ($\xi = 2.4$), o ganho Proporcional ($K$) é escalonado utilizando a proporção $\eta = u / u_n$:
$$K = \frac{K_{nom}}{\eta^2}$$

### 2.2 Equações a Diferenças (Filtro Digital)
As funções de transferência são discretizadas a $1s$ (Zero-Order Hold). A classe `ControladorIara` atua como filtro retendo valores passados ($k-1, k-2$) para suavizar a entrada em degrau da posição e gerar o ângulo do leme ($s_{ang}$) final:
$$s_{ang}[k] = \alpha_1 s_{ang}[k-1] + \alpha_2 s_{ang}[k-2] + \beta_1 i_{ang}[k-1] + \beta_2 i_{ang}[k-2]$$

---

## 3. Cinemática no Simulador (`runner.py`)

Em vez de interpolação linear, a movimentação do barco ocorre através de **cálculo cinemático em tempo real** a cada passo $\Delta t = 0.05s$ de simulação.

### 3.1 Ângulo de Referência
O erro angular ($i_{ang}$) que alimenta o controlador é calculado a cada frame pela função arco-tangente entre o destino e o barco:
$$i_{ang} = \text{atan2}(y_{fim} - y_{atual}, x_{fim} - x_{atual})$$

### 3.2 Integração do Movimento Uniforme
O barco gira seu nariz fisicamente para o ângulo ditado pelo controlador ($s_{ang}$) e avança no espaço 3D usando as componentes vetoriais da sua velocidade parametrizada:
$$X_{novo} = X_{atual} + (v \cdot \cos(s_{ang}) \cdot \Delta t)$$
$$Y_{novo} = Y_{atual} + (v \cdot \sin(s_{ang}) \cdot \Delta t)$$
