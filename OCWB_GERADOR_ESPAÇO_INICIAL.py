# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:11:14 2016

@author: Nemesio Sopelsa
"""
def OGEI(NNG,NCC,NEL,NGE,CNI,LXCC,LYCC,SCC,TME,TEE):
    import numpy as np
    """NNCC é um indicador quais nós estão no contorno (Desnecessário)
    PGNCC indica a mesma coisa que NNCC só que diferenciando o numero de cada nós no contorno
    SSCC indica o valor pontual para cada nó indicado em PGNCC 
    NTFC indica quantos nós possuem condição de contorno (A principio desnecessário mas serve como aux para PGNCC)"""
    NNCC = np.zeros((NNG,1))
    PGNCC = np.zeros((NNG,1))
    SSCC = np.array([])
    NTFC = 0
    """for i e for j testam cada nó da nossa malha para ver se eles estão contidos ou não nas condições de contorno"""
    for i in range(NNG):
        for j in range(NCC):
            if CNI[i][0] >= LXCC[j][0] and CNI[i][0] <= LXCC[j][1] and CNI[i][1] >= LYCC[j][0] and CNI[i][1] <= LYCC[j][1]:
                NNCC[i] = 1
                NTFC += 1
                PGNCC[i] = NTFC 
                SSCC=np.append(SSCC,SCC[j])
                break
    """IFBG = vetor fonte
    IFBEG = matriz de rigidez 
    GFBE e FBE são vetores pré determidados que serão necessários para o calculo da interação de cada elemento"""
    IFBG = np.zeros((NNG,NNG))
    IFBEG = np.zeros((NNG,1))
    GFBE = np.array([[-1,1,0],[-1,0,1]])
    FBE = np.array([1-1/3-1/3,1/3,1/3])
    CXNE=np.zeros([NEL,6])
    CYNE=np.zeros([NEL,6])
    for i in range(NEL):
        for j in range(3):
            """Para cada nó de cada elemento testado nesse for i vou criar um indicador "IND" de quais nós formam meu elemento, 
            que a principio é desnecessario além de criar variaveis CXNE e CYNE que indicaram todos os x e y de cada nó do meu 
            triangulo"""
            IND=int(NGE[i][j])
            CXNE[i][j]=CNI[IND][0]
            CYNE[i][j]=CNI[IND][1]
        """Além de marcar (nas posições 0,1 e 2) as três coordenas x e y de cada um dos meus três pontos, é criado três outras
        posições no vetor CXNE que indicarão o ponto medio entre [(x1 e x2),(x1 e x3),(x2 e x3)], sendo feito de forma
        equivalente para o y"""
        CXNE[i][3]=(CXNE[i][0]+CXNE[i][1])/2
        CXNE[i][4]=(CXNE[i][0]+CXNE[i][2])/2
        CXNE[i][5]=(CXNE[i][1]+CXNE[i][2])/2
        CYNE[i][3]=(CYNE[i][0]+CYNE[i][1])/2
        CYNE[i][4]=(CYNE[i][0]+CYNE[i][2])/2
        CYNE[i][5]=(CYNE[i][1]+CYNE[i][2])/2
        """Com minha coordenadas x e y em mão, posso calcular minha componente jacobiana, e realizar as operações necessárias
        para o calculo da contribuição daquele elemento para matriz de rigidez e vetor fonte, perceba que o tipo de material
        influencia na minha matriz de rigidez e o tipo de excitação influencia no meu vetor fonte"""
        J = np.array([[CXNE[i][1]-CXNE[i][0],CYNE[i][1]-CYNE[i][0]],[CXNE[i][2]-CXNE[i][0],CYNE[i][2]-CYNE[i][0]]])
        GFBET = np.linalg.solve(J,GFBE)
        IGFBE = np.transpose(GFBET).dot(GFBET)*TME[i]*np.abs(np.linalg.det(J))*0.5
        IFBE = FBE*TEE[i]*np.abs(np.linalg.det(J))*0.5
        """Agora tomando pelo critério da superposição eu vou jogando a influencia de cada um dos meus nós nas minhas matrizes
        principais"""
        for j in range(3):
            IFBEG[int(NGE[i][j])] += IFBE[j]
            for k in range(3):
                IFBG[int(NGE[i][j])][int(NGE[i][k])] += IGFBE[j][k]
    """Depois de ter criado minha matriz de rigidez e vetor de fonte eu aplico sobre ela as condições de contorno de Dirichlet"""
    for i in range(NNG):
        if NNCC[i] == 1:
            IFBG[i] = 0
            IFBG[i][i] = 1
            IFBEG[i] = SSCC[int(PGNCC[i]-1)]
    """Com minhas condições de contorno aplicadas eu posso calcular a solução inicial da minha malha pelo metodo tradicional dos
    elementos finitos, usando a função solve que inverte minha matriz IFBEG e multiplica por IFBG"""
    SI = np.linalg.solve(IFBG,IFBEG)
    return NNCC,PGNCC,SSCC,NTFC,IFBG,IFBEG,GFBE,FBE,CXNE,CYNE,J,GFBET,IGFBE,IFBE,SI