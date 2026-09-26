# Gain-Scheduling-Simulation
Simulação no Coppelia Sim de um algoritmo de Gain Scheduling

# Equações e Funções utilizadas no _runner.py_

## 1.Função geométrica de mapeamento:
$$ d = \sqrt{(x_b - x_a)^2 + (y_b-y_a)^2}$$ 

esta equação do teorema de Pitágoras diz o quão longe o objeto, no caso o barco chamado IARA, está distante do seu objetivo final 


## 2. Função _input_de_ponto( )_

Essa função recebe uma string que é captada pelo input do usuário no terminal, a string enviada no formato "x,y", por exemplo "10,10", e utilizando o "." como separador de decimais nos números float.

## 3. Função _posicao(barco_handle)_

Função de telemetria. Consulta a API do CoppeliaSim para extrair a coordenada espacial absoluta $[X, Y, Z]$ do centro de massa do modelo do barco em relação ao referencial do mundo simulado.

## 4. Função _obstaculos_cena(nome_obstaculos, margem)_


Realiza o mapeamento do cenário. Varre o CoppeliaSim buscando as posições absolutas das plantas a partir dos seus determinados nomes na cena e cria "zonas de exclusão" circulares. Aplicamos o conceito de Espaço de Configuração (C-Space). Como o barco é tratado como uma partícula pontual no codigo de controle _controle_iara.py_, adicionei aos obstáculos uma margem de segurança ao raio físico: 

$$R_{total} = R_{planta} + \text{margem}$$

Isso nos garante que se a "partícula" passar muito próximo no círculo matemático de $R_{total}$, o casco real do barco não encostará no obstaculo.

## 5. Algoritmo RRT (Planejamento de Trajetória) -> _ponto_aleatorio()_ 

Amostra o espaço 2D utilizando uma distribuição de probabilidade uniforme contínua. Para os limites configurados, ele sorteia variáveis independentes:

$$x \sim U(X_{MIN}, X_{MAX})$$

$$y \sim U(Y_{MIN}, Y_{MAX})$$

## 6. _ponto_valido(p, obstaculos)_

Resolve uma Inequação de Círculo para garantir segurança. Um ponto $(x,y)$ só é válido se estiver dentro dos limites do mapa e fora de todas as zonas de exclusão. A condição matemática para rejeitar o ponto se ele estiver dentro do obstáculo $i$ é:

$$(x - x_i)^2 + (y - y_i)^2 \leq R_i^2$$

## 7. _colide_segmento(a, b, obstaculos, resolucao)_

Resolve o problema de interceptação geométrica discretizando uma reta. Aplica a Equação Paramétrica da Reta para criar "passos" entre o ponto $A$ e o ponto $B$:

$$P(t) = A + t(B - A)$$

Onde $t$ varia de $0$ a $1$. O código avança a variável $t$ em pequenas frações de passo (resolucao) e checa se algum desses pontos $P(t)$ intermediários cai dentro da inequação de um obstáculo. Se cair, a linha reta cruza uma planta.

## 7. _no_mais_proximo(arvore, ponto)_ 

Realiza uma busca por vizinho mais próximo (Nearest Neighbor) no espaço vetorial. Varre todos os nós já criados na árvore e retorna o índice daquele que minimiza a distância Euclidiana até o novo ponto sorteado:

$$\text{Índice} = \arg\min_{i \in \text{arvore}} \Vert{} P_i - P_{novo} \Vert{}_2$$

## 8. _steer(origem, destino, passo)_ 

Limita o crescimento da árvore. Se o ponto sorteado estiver muito longe da origem, esta função satura o vetor de deslocamento. Ela calcula o vetor direção, normaliza para criar um vetor unitário, e o multiplica pelo tamanho máximo permitido (passo). Seja o vetor direção $\vec{v} = (x_{destino} - x_{origem}, y_{destino} - y_{origem})$

Distância (módulo): $d = \Vert{} \vec{v} \Vert{}$

Componentes normalizados: $dx = \frac{v_x}{d}$ e $dy = \frac{v_y}{d}$

Novo ponto gerado: $P_{novo} = P_{origem} + \text{passo} \cdot (dx, dy)$

## 9. _reconstruir_caminho(arvore, indice_final)_

Executa um Backtracking clássico em grafos. Começa no nó que atingiu o objetivo e segue a lista de ponteiros pai iterativamente até chegar ao nó raiz (pai: None). Depois inverte a lista resultante para entregar o caminho da origem ao destino.

## 10. _RRT(posicao_inicial, objetivo, obstaculos)_ 

O núcleo do espaço de estados. A cada iteração: Aplica o GOAL_BIAS: Com 10% de probabilidade impõe o destino como ponto aleatório (direcionando o crescimento) ou sorteia no espaço com 90%. Acha o nó mais próximo da árvore. Cria um novo ponto usando steer. Valida as restrições geométricas com ponto_valido e colide_segmento. Se o novo ponto se conectar sem colisões ao destino, encerra a busca. Visualização e Execução Cinemática

## 11. _visualizar_rrt(...)_ 

Módulo de renderização gráfica. Usa os vetores de posições guardados nos dicionários da árvore para desenhar retas sobrepostas no Matplotlib usando funções de desenho de formas (Patches) e plotagem linear, com plt.pause() travando a thread para renderizar frame a frame.

## 12. Matemática de Atuação no Loop Principal (Execução ponto a ponto) 

dentro do laço _for i in range(len(caminho) - 1):_, duas operações físicas cruciais ocorrem:

Cálculo de Rotação (Yaw):

Para rotacionar o barco no eixo Z apontando seu nariz para o próximo waypoint, utiliza-se a função trigonométrica arco-tangente2, que resolve a ambiguidade de quadrantes dividindo $\Delta y$ por $\Delta x$.

$$\theta_{yaw} = \text{atan2}(y_{fim} - y_{inicio}, x_{fim} - x_{inicio}) + \pi$$

(A constante $\pi$ em radianos corresponde a 180°, usada para inverter o eixo referencial do modelo 3D).

Interpolação de Movimento Controlado:

Em vez de teleportar o barco a velocidades constantes, a posição instantânea é modulada pela resposta transitória da sua função de transferência discreta gerada pela biblioteca control. O fator é a amplitude da resposta ao degrau no instante $t$ (variando de 0 a 1).A equação é uma interpolação linear parametrizada não pelo tempo real, mas pela resposta do controlador $C(t)$:
