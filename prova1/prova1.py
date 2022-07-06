from cProfile import label
from re import T
import statistical_module as sm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
######################################################
# ELISA RODRIGUES DE SOUSA COUTINHO - 11811FMD021
# RICARDO TADEU OLIVEIRA CATTA PRETA - 11911FMT028
# ROGÉRIO BUSO DE ANDRADE - 12011FMT009
# THAYNÁ CAROLINE SABINO DE ASSUMPÇÃO - 11811FMD013
# Laboratório de Física Básica 2
# ------------Pêndulo simples------------------------------
#######################################################

# Importa os dados coletados do experimento com uma mola, considerando a primeira massa
pendulo = pd.read_csv("pendulo_prova.csv", keep_default_na=True) #importa a lista de dados

def mola_serie():
    """
    Função para plotar os gráficos do tempo em função da massa, juntamente com 
    seus valores linearizados.
    """
    t1 = pendulo.t1    
    t2 = pendulo.t2 
    t3 = pendulo.t3
    t4 = pendulo.t4    
    t5 = pendulo.t5 
    t6 = pendulo.t6
    t7 = pendulo.t7
    t8 = pendulo.t8

    l1 = pendulo.l1    
    l2 = pendulo.l2 
    l3 = pendulo.l3
    l4 = pendulo.l4    
    l5 = pendulo.l5 
    l6 = pendulo.l6 
    l7 = pendulo.l7 
    l8 = pendulo.l8

    t = np.array([t1, t2, t3, t4, t5, t6, t7, t8])
    l = np.array([l1, l2, l3, l4, l5, l6, l7, l8])
    print("t = ", t)
    print("l = ", l)

    #Cálculo do erro associado para o tempo
    t_ea = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005,0.0005,0.0005, 0.0005])

    #Cálculo do erro associado para a massa
    l_ea = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005,0.0005,0.0005, 0.0005])

    print("t_ea = ", t_ea)
    print("l_ea = ", l_ea)

    #linearizando os resultados
    ln_t = np.log10(t)
    ln_l = np.log10(l)
    print("ln_l = \n", ln_l)
    print("ln_t = \n", ln_t)
    print("ln_t*ln_l = \n", ln_t*ln_l)
    print("ln_l*ln_l = \n", ln_l*ln_l)

    #Cálculo dos coeficientes linear e angular, calculado pelo MMQ
    m1, b1 = sm.least_square(ln_l, ln_t, 4)
    y1 = m1 * ln_l + b1

    print("coeoficiente angular m = ", m1)
    print("coeoficiente linear b = ", b1)

    erro_asso_t = sm.erro_associado(t, 4, 0.0005)
    erro_asso_l = sm.erro_associado(l, 4, 0.0005)

    propaga_inc_t0= sm.propaga_incerteza_3D(1/t[0], 0,0,t_ea[0], 0, 0)
    propaga_inc_t1= sm.propaga_incerteza_3D(1/t[1], 0,0,t_ea[1], 0, 0)
    propaga_inc_t2= sm.propaga_incerteza_3D(1/t[2], 0,0,t_ea[2], 0, 0)
    propaga_inc_t3= sm.propaga_incerteza_3D(1/t[3], 0,0,t_ea[3], 0, 0)
    propaga_inc_t4= sm.propaga_incerteza_3D(1/t[4], 0,0,t_ea[4], 0, 0)
    propaga_inc_t5= sm.propaga_incerteza_3D(1/t[5], 0,0,t_ea[5], 0, 0)
    propaga_inc_t6= sm.propaga_incerteza_3D(1/t[6], 0,0,t_ea[6], 0, 0)
    propaga_inc_t7= sm.propaga_incerteza_3D(1/t[7], 0,0,t_ea[7], 0, 0)

    print("\n propaga_inc_t0 = ", propaga_inc_t0)
    print("\n propaga_inc_t1 = ", propaga_inc_t1)
    print("\n propaga_inc_t2 = ", propaga_inc_t2)
    print("\n propaga_inc_t3 = ", propaga_inc_t3)
    print("\n propaga_inc_t4 = ", propaga_inc_t4)
    print("\n propaga_inc_t5 = ", propaga_inc_t5)
    print("\n propaga_inc_t6 = ", propaga_inc_t6)
    print("\n propaga_inc_t7 = ", propaga_inc_t7)

    propaga_inc_l0= sm.propaga_incerteza_3D(1/l[0], 0,0,l_ea[0], 0, 0)
    propaga_inc_l1= sm.propaga_incerteza_3D(1/l[1], 0,0,l_ea[1], 0, 0)
    propaga_inc_l2= sm.propaga_incerteza_3D(1/l[2], 0,0,l_ea[2], 0, 0)
    propaga_inc_l3= sm.propaga_incerteza_3D(1/l[3], 0,0,l_ea[3], 0, 0)
    propaga_inc_l4= sm.propaga_incerteza_3D(1/l[4], 0,0,l_ea[4], 0, 0)
    propaga_inc_l5= sm.propaga_incerteza_3D(1/l[5], 0,0,l_ea[5], 0, 0)
    propaga_inc_l6= sm.propaga_incerteza_3D(1/l[6], 0,0,l_ea[6], 0, 0)
    propaga_inc_l7= sm.propaga_incerteza_3D(1/l[7], 0,0,l_ea[7], 0, 0)

    propaga_inc_lnl = np.array([propaga_inc_l0, propaga_inc_l1, propaga_inc_l2, 
    propaga_inc_l3, propaga_inc_l4, propaga_inc_l5, propaga_inc_l6, propaga_inc_l7])

    propaga_inc_lnt = np.array([propaga_inc_t0, propaga_inc_t1, propaga_inc_t2, 
    propaga_inc_t3, propaga_inc_t4, propaga_inc_t5, propaga_inc_t6, propaga_inc_t7])

    print("\n propaga_inc_l0 = ", propaga_inc_l0)
    print("\n propaga_inc_l1 = ", propaga_inc_l1)
    print("\n propaga_inc_l2 = ", propaga_inc_l2)
    print("\n propaga_inc_l3 = ", propaga_inc_l3)
    print("\n propaga_inc_l4 = ", propaga_inc_l4)
    print("\n propaga_inc_l5 = ", propaga_inc_l5)
    print("\n propaga_inc_l6 = ", propaga_inc_l6)
    print("\n propaga_inc_l7 = ", propaga_inc_l7)

    plt.style.use('ggplot')
    fig = plt.figure(dpi=130)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{t}$ $[s]$')
    axes1.set_xlabel('$\ln{l}$ $[m]$')
    #axes1.set_ylabel('${t}$ $[s]$')
    #axes1.set_xlabel('${l}$ $[m]$')
    #plt.plot(l, t, '-', label="L x T")
    plt.plot(ln_l, y1, '-', label="regressão linear por MMQ")
    #Coloca a barra de erro%
    ls = ''

    #print("\n incerteza de m = \n", sm.sigma_m(erro_asso_t ,ln_m))
    #print("\n incerteza de b = \n", sm.sigma_b(erro_asso_t ,ln_m))

    gravidade = ((2 * np.pi) / (10**(b1))) ** 2
    print("gravidade = \n", gravidade)

    dg_dl = ((2 * np.pi) / np.mean(t)) ** 2
    delta_l = 0.0005
    dg_dt = -((8 * (np.pi ** 2) * np.mean(l))  / (np.mean(t) ** 3))
    delta_t = 0.0005
    delta_g = np.sqrt((dg_dl * delta_l) ** 2 + (dg_dt * delta_t) ** 2)
    print("delta_gravidade = \n", delta_g)

    #for i in range(len(t)):
    #    if i == 0:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l1")  
    #    elif i == 1:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l2")
    #    elif i == 2:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l3")
    #    elif i == 3:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l4")
    #    elif i == 4:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l5")
    #    elif i == 5:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l6")
    #    elif i == 6:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l7")    
    #    else:
    #        plt.errorbar(l[i], t[i], xerr=l_ea[i],  yerr=t_ea[i], linestyle=ls, marker='o',label ="ponto experimental l8")

    for i in range(len(t)):
        if i == 0:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l1")  
        elif i == 1:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l2")
        elif i == 2:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l3")
        elif i == 3:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l4")
        elif i == 4:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l5")
        elif i == 5:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l6")
        elif i == 6:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l7")            
        else:
            plt.errorbar(ln_l[i], ln_t[i], xerr=propaga_inc_lnl[i],  yerr=propaga_inc_lnt[i], linestyle=ls, marker='o',label ="ponto experimental l8")
    
    fig.tight_layout()
    plt.title("Pêndulo simples")
    plt.legend(loc='best')
    plt.show()

mola_serie()
