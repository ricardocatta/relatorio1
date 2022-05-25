from cProfile import label
from re import T
import statistical_module as sm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
######################################################
# Ricardo Tadeu Oliveira Catta Preta
# matrícula: 11911FMT028
# Laboratório de Física Básica 2
######################################################

# Importa os dados coletados do experimento com uma mola, considerando a primeira massa
umamola = pd.read_csv("mola_massa1.csv", keep_default_na=True) #importa a lista de dados
print("umamola = \n", umamola)

# Importa os dados coletados do experimento com uma mola, considerando a segunda massa
umamola_massa2 = pd.read_csv("mola_massa2.csv", keep_default_na=True) 
print("umamola_massa2 = \n", umamola_massa2)

# Importa os dados coletados do experimento com uma mola, considerando a terceira massa
umamola_massa3 = pd.read_csv("mola_massa3.csv", keep_default_na=True) 
print("umamola_massa3 = \n", umamola_massa3)

# Importa os dados coletados do experimento com duas molas em paralelo, 
# considerando a primeira massa
paralelo_massa1 = pd.read_csv("paralelo_massa1.csv", keep_default_na=True) 

# Importa os dados coletados do experimento com duas molas em paralelo, 
# considerando a segunda massa
paralelo_massa2 = pd.read_csv("paralelo_massa2.csv", keep_default_na=True) 

# Importa os dados coletados do experimento com duas molas em paralelo, 
# considerando a segunda massa
paralelo_massa3 = pd.read_csv("paralelo_massa3.csv", keep_default_na=True) 

def mola_serie():
    """
    Função para plotar os gráficos do tempo em função da massa, juntamente com 
    seus valores linearizados
    """
    m1 = np.mean(umamola.m1)/ 1000    
    m2 = np.mean(umamola_massa2.m2) / 1000 
    m3 = np.mean(umamola_massa3.m3)/ 1000  
    m = np.array([m1, m2, m3])
    print("m = ", m)

    t1 = np.mean(umamola.t1)  / 10
    t2 = np.mean(umamola_massa2.t2) / 10
    t3 = np.mean(umamola_massa3.t3) / 10

    t = np.array([t1, t2, t3])
    print("tempo em série = \n", t)

    #Cálculo do erro associado para o tempo
    t_ea = np.array([sm.erro_associado(umamola.t1, 4, 0.0005), sm.erro_associado(umamola_massa2.t2, 4, 0.0005), 
    sm.erro_associado(umamola_massa3.t3, 4, 0.0005)])

    #Cálculo do erro associado para a massa
    m_ea = np.array([sm.erro_associado(umamola.m1, 4, 0.0005), sm.erro_associado(umamola_massa2.m2, 4, 0.0005), 
    sm.erro_associado(umamola_massa3.m3, 4, 0.0005)])
    print("t_ea = ", t_ea)
    print("m_ea = ", m_ea)

    #linearizando os resultados
    ln_t = np.log(t)
    ln_m = np.log(m)
    print("ln_m = \n", ln_m)
    print("ln_t = \n", ln_t)
    print("ln_t*ln_m = \n", ln_t*ln_m)
    print("ln_m*ln_m = \n", ln_m*ln_m)

    #Cálculo dos coeficientes linear e angular, calculado pelo MMQ
    m1, b1 = sm.least_square(ln_m, ln_t, 5)
    y1 = m1 * ln_m + b1

    print("coeoficiente angular m = ", m1)
    print("coeoficiente linear b = ", b1)

    erro_asso_t = sm.erro_associado(t, 4, 0.0005)
    erro_asso_m = sm.erro_associado(m, 4, 0.0005)

    propaga_inc_m0= sm.propaga_incerteza_3D(1/m[0], 0,0,erro_asso_m, 0, 0)
    propaga_inc_m1= sm.propaga_incerteza_3D(1/m[1], 0,0,erro_asso_m, 0, 0)
    propaga_inc_m2= sm.propaga_incerteza_3D(1/m[2], 0,0,erro_asso_m, 0, 0)

    print("\n propaga_inc_m0 = ", propaga_inc_m0)
    print("\n propaga_inc_m1 = ", propaga_inc_m1)
    print("\n propaga_inc_m2 = ", propaga_inc_m2)

    propaga_inc0_t0= sm.propaga_incerteza_3D(0, 1/t[0],0, 0, erro_asso_t, 0)
    propaga_inc1_t1= sm.propaga_incerteza_3D(0, 1/t[1],0, 0, erro_asso_t, 0)
    propaga_inc2_t2= sm.propaga_incerteza_3D(0, 1/t[2],0, 0, erro_asso_t, 0)

    print("\n propaga_inc_t0 = ", propaga_inc0_t0)
    print("\n propaga_inc_t1 = ", propaga_inc1_t1)
    print("\n propaga_inc_t2 = ", propaga_inc2_t2)

    plt.style.use('ggplot')
    fig = plt.figure(dpi=130)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('${t}$ $[s]$')
    axes1.set_xlabel('${m}$ $[kg]$')
    #plt.plot(m , t, '*', label="t x m")
    #plt.plot(ln_m, y1, '-', label="regressão linear por MMQ")
    #Coloca a barra de erro%
    ls = ''

    print("\n incerteza de m = \n", sm.sigma_m(erro_asso_t ,ln_m))
    print("\n incerteza de b = \n", sm.sigma_b(erro_asso_t ,ln_m))

    k1_mola = ((2 * np.pi) / (np.exp(b1))) ** 2
    print("k1_mola = \n", k1_mola)

    dk_dm = 1 / np.mean(m)
    delta_m = np.mean(m_ea)
    dk_dt = 1 / np.mean(t)
    delta_t = np.mean(t_ea)
    delta_k = k1_mola * np.sqrt((dk_dm * delta_m) ** 2 + (2 * dk_dt * delta_t) ** 2)
    print("delta_k = \n", delta_k)



    #for i in range(len(t)):
    #    if i == 0:
    #        plt.errorbar(ln_m[i], ln_t[i], xerr=propaga_inc_m0,  yerr=propaga_inc0_t0, linestyle=ls, marker='o',label ="ponto experimental para m1")  
    #    elif i == 1:
    #        plt.errorbar(ln_m[i], ln_t[i], xerr=propaga_inc_m1,  yerr=propaga_inc1_t1, linestyle=ls, marker='o',label ="ponto experimental para m2")
    #    elif i == 2:
    #        plt.errorbar(ln_m[i], ln_t[i], xerr=propaga_inc_m2,  yerr=propaga_inc2_t2, linestyle=ls, marker='o',label ="ponto experimental para m3")
    #    else:
    #        plt.errorbar(ln_t[i], ln_t[i], xerr=propaga_inc_m2[i],  yerr=propaga_inc_m2[i], linestyle=ls, marker='o',label ="pontos experimentais P7")

    for i in range(len(t)):
        if i == 0:
            plt.errorbar(m[i], t[i], xerr=m_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental para m1")  
        elif i == 1:
            plt.errorbar(m[i], t[i], xerr=m_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental para m2")
        elif i == 2:
            plt.errorbar(m[i], t[i], xerr=m_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental para m3")
        else:
            plt.errorbar(m[i], t[i], xerr=m_ea[i],  yerr=t_ea[i][i], linestyle=ls, marker='o',label ="pontos experimentais P7")

    fig.tight_layout()
    plt.title("Uma mola")
    plt.legend(loc='best')
    plt.show()




def mola2_paralelo():
    """
    Função para plotar os gráficos do tempo em função da massa, juntamente com 
    seus valores linearizados
    """
    
    mp1 = np.mean(paralelo_massa1.mp1)/ 1000    
    mp2 = np.mean(paralelo_massa2.mp2) / 1000 
    mp3 = np.mean(paralelo_massa3.mp3)/ 1000  
    mp = np.array([mp1, mp2, mp3])
    print("mp = ", mp)

    tp1 = np.mean(paralelo_massa1.tp1) / 10
    tp2 = np.mean(paralelo_massa2.tp2) / 10
    tp3 = np.mean(paralelo_massa3.tp3) / 10

    tp = np.array([tp1, tp2, tp3])

    #Cálculo do erro associado para o tempo
    tp_ea = np.array([sm.erro_associado(paralelo_massa1.tp1, 4, 0.0005), 
    sm.erro_associado(paralelo_massa2.tp2, 4, 0.0005), 
    sm.erro_associado(paralelo_massa3.tp3, 4, 0.0005)])

    #Cálculo do erro associado para a massa
    mp_ea = np.array([sm.erro_associado(paralelo_massa1.mp1, 4, 0.0005), 
    sm.erro_associado(paralelo_massa2.mp2, 4, 0.0005), 
    sm.erro_associado(paralelo_massa3.mp3, 4, 0.0005)])
    print("tp_ea = ", tp_ea)
    print("mp_ea = ", mp_ea)
    print("tp = ", tp)

    #linearizando os resultados
    ln_t = np.log(tp)
    ln_m = np.log(mp)
    print("ln_mp = \n", ln_m)
    print("ln_tp = \n", ln_t)
    print("ln_tp*ln_mp = \n", ln_t*ln_m)
    print("ln_mp*ln_mp = \n", ln_m*ln_m)

    #Calculando os coeficientes lineares e angulares pelo método dos MMQ
    m1, b1 = sm.least_square(ln_m, ln_t, 5)
    y1 = m1 * ln_m + b1

    print("coeficiente angular mp = ", m1)
    print("coeficiente linear bp = ", b1)

    #Cálculo do erro associado para o tempo e para massa
    erro_asso_t = sm.erro_associado(tp, 4, 0.0005)
    erro_asso_m = sm.erro_associado(mp, 4, 0.0005)
    propaga_inc_m0= sm.propaga_incerteza_3D(1/mp[0], 0,0,erro_asso_m, 0, 0)
    propaga_inc_m1= sm.propaga_incerteza_3D(1/mp[1], 0,0,erro_asso_m, 0, 0)
    propaga_inc_m2= sm.propaga_incerteza_3D(1/mp[2], 0,0,erro_asso_m, 0, 0)

    print("\n propaga_inc_m0 = ", propaga_inc_m0)
    print("\n propaga_inc_m1 = ", propaga_inc_m1)
    print("\n propaga_inc_m2 = ", propaga_inc_m2)

    propaga_inc0_t0= sm.propaga_incerteza_3D(0, 1/tp[0],0, 0, erro_asso_t, 0)
    propaga_inc1_t1= sm.propaga_incerteza_3D(0, 1/tp[1],0, 0, erro_asso_t, 0)
    propaga_inc2_t2= sm.propaga_incerteza_3D(0, 1/tp[2],0, 0, erro_asso_t, 0)

    print("\n propaga_inc_t0 = ", propaga_inc0_t0)
    print("\n propaga_inc_t1 = ", propaga_inc1_t1)
    print("\n propaga_inc_t2 = ", propaga_inc2_t2)

    plt.style.use('ggplot')
    fig = plt.figure(dpi=130)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{t}$ $[s]$')
    axes1.set_xlabel('$\ln{m}$ $[kg]$')
    #plt.plot(m , t, '*', label="t x m")
    plt.plot(ln_m, y1, '-', label="regressão por MMQ para as molas em paralelo")
    #Coloca a barra de erro%
    ls = ''

    print("\n incerteza de mp = \n", sm.sigma_m(erro_asso_t ,ln_m))
    print("\n incerteza de bp = \n", sm.sigma_b(erro_asso_t ,ln_m))

    k1_mola = ((2 * np.pi) / (np.exp(b1))) ** 2
    print("k1_mola_paralelo = \n", k1_mola)

    dk_dm = 1 / np.mean(mp)
    delta_m = np.mean(mp_ea)
    dk_dt = 1 / np.mean(tp)
    delta_t = np.mean(tp_ea)
    delta_k = k1_mola * np.sqrt((dk_dm * delta_m) ** 2 + (2 *dk_dt * delta_t) ** 2)
    print("delta_k_paralelo = \n", delta_k)



    for i in range(len(tp)):
        if i == 0:
            plt.errorbar(ln_m[i], ln_t[i], xerr=propaga_inc_m0,  yerr=propaga_inc0_t0, linestyle=ls, marker='o',label ="pontos experimentais para m1")  
        elif i == 1:
            plt.errorbar(ln_m[i], ln_t[i], xerr=propaga_inc_m1,  yerr=propaga_inc1_t1, linestyle=ls, marker='o',label ="pontos experimentais para m2")
        elif i == 2:
            plt.errorbar(ln_m[i], ln_t[i], xerr=propaga_inc_m2,  yerr=propaga_inc2_t2, linestyle=ls, marker='o',label ="pontos experimentais para m3")
        else:
            plt.errorbar(ln_t[i], ln_t[i], xerr=propaga_inc_m2[i],  yerr=propaga_inc_m2[i], linestyle=ls, marker='o',label ="pontos experimentais P7")  
    #for i in range(len(tp)):
    #    if i == 0:
    #        plt.errorbar(mp[i], tp[i], xerr=mp_ea[i],  yerr=tp_ea[i], linestyle=ls, marker='o',label ="ponto experimental para m1")  
    #    elif i == 1:
    #        plt.errorbar(mp[i], tp[i], xerr=mp_ea[i],  yerr=tp_ea[i], linestyle=ls, marker='o',label ="ponto experimental para m2")
    #    elif i == 2:
    #        plt.errorbar(mp[i], tp[i], xerr=mp_ea[i],  yerr=tp_ea[i], linestyle=ls, marker='o',label ="ponto experimental para m3")
    #    else:
    #        plt.errorbar(mp[i], tp[i], xerr=mp_ea[i],  yerr=tp_ea[i][i], linestyle=ls, marker='o',label ="pontos experimentais P7")

    fig.tight_layout()
    plt.title("Duas molas em paralelo")
    plt.legend(loc='best')
    plt.show()

mola_serie()
mola2_paralelo()