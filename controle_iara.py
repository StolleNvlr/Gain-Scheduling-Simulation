import control as ct
import math as m
import matplotlib.pyplot as plt
'''
Definição de variaveis globais, funções de transferencia e valores iniciais
'''
a_n = 1.6
b_n = 0.2
velocidade = 1.945 #nós
s = ct.tf('s') #função de transferencia
'''
Definição de parametros do controlador e da planta
'''

K= 30
Td = (2.4*m.sqrt(0.2*K)-1.6)/(0.2*K);
'''
Definição de variaveis parametrizadas
'''
v = 10
a_param = a_n*(velocidade/v)
b_param = b_n*(velocidade/v)**2
'''
Definição do controlador
'''
G = (K*b_param*(1+s*Td))/(s**2 + s*(a_param+K*b_param*Td) + K*b_param); #controlador
'''
Definição de função de conversão de tempo continuo para discreto
'''
G_discreto = ct.c2d(G,0.05) #tempo discreto com amostragem de 1 segundo

'''
Definição de graficos para visualização
'''
plt.clf()
plt.figure(1)
def dinamica():
    tempo, resposta = ct.step_response (G_discreto)

    #plt.step(tempo,resposta, where='post', color='#0072BD', linewidth = 1)
    #plt.title('Step Response')
    #plt.grid(False)
    #plt.xlim([0,20])
    #plt.ylim([0, 1])
    #y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #y_labels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    #plt.yticks(y_ticks, y_labels)
    #plt.show()
    return tempo, resposta




