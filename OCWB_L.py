# -*- coding: utf-8 -*-       

"""tic e toc são funções que calculam o tempo que o tempo demora para rodar"""

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")


import matplotlib.pylab as plt
import scipy.sparse as sps
import numpy as np
import FEM_GERADOR_MALHA_PY as FGMP
import OCWB_GERADOR_ESPAÇO_INICIAL as OGEI
import OCWB_PROCESSADOR_L as OPL
import L
tic()


"""APOSTERIORI e APRIORI são dois metodos diferentes de refinamento que o programa aceita, colocando um dos dois em 1 e outro em 0
para descedir qual utilizar. A variavel LDET indica o quanto de erro é aceito e o programa refina de acordo com os pontos aonde tem
maior erro, já o LX1,LX2,LY1 e LY2 indicam a cada interação quais são as áreas que terão prioridade no programa
NI indica o numero de vezes que o programa vai refinar"""

APOSTERIORI = 1
LDET = 0.005
APRIORI = 0
LX1 = np.array([0,0.25,0.25,0.25,0.25])
LX2 = np.array([1,0.75,0.75,0.75,0.75])
LY1 = np.array([0,0.25,0.25,0.25,0.25])
LY2 = np.array([1,0.75,0.75,0.75,0.75])
NI = 1

"""A função L determina nosso domínio de estudo, nossa função FGMP cria nossa malha inicial, nossa função OGEI gera nossa primeira
solução mais grosseira do problema pelo metodo tradicional dos elementos finitos e nosso OPL gera as primeiras wavelets em cima de
nossa malha criada"""
LXDOM,LYDOM,LSI,LXCC,LYCC,SCC,LXMAT,LYMAT,VMAT,LXEXC,LYEXC,VEXC,NDOM,NCC,NMAT,NEXC = L.L()
CXNI,CYNI,INDNGE,INDCNI,IXPN,IYPN,NNYDOM,NNXDOM,AEEW,CNI,NEL,CNIL,NGE,NMN,NNG,TE,TEE,TME,IND1,IND,PEEX,PEEY,INDA = FGMP.FGMP()
NNCC,PGNCC,SSCC,NTFC,IFBG,IFBEG,GFBE,FBE,CXNE,CYNE,J,GFBET,IGFBE,IFBE,SI = OGEI.OGEI(NNG,NCC,NEL,NGE,CNI,LXCC,LYCC,SCC,TME,TEE)
EFN2NC,EFN1NC,REES,RNE,PPNTE,FBEL,GFBEL,TEES,ZZZ,TN,T1,T2,T3,T4,T5,NFTN,PWTN,NGTN,NFN = OPL.OPL(NNG,NMN,CNI)


"""SOCWB = Matriz RESPOSTA com os valores dos potenciais de cada ponto
COCWB = posição [x,y] de cada ponto de nossa malha
PNECNF = matriz que indica quais pontos formam cada um de nossos elementos da malha
MHAP = nós em que será feito o refinamento na próxima etapa
MAA = 
"""

SOCWB = SI
COCWB = np.array(CNI)
MHAP = np.ones((NNG,1))
PNECNF = np.zeros((NEL,6))
PNECNF[:,0:3] = np.array(NGE)
MAA = np.ones((NEL,1))

"""kl, con9, LENECI, PNCNF, PNECNFN são somente auxiliares que serão usados posteriormente
TEN = 
TMEN =
TEEN =
MAAN =
ECI =
NFSNC = numero de funções que faltam para fechar a malha inicial para o caso L
NWP = irá indicar em quais nós sera necessário adicionar quantos funções especiais de wavelets 
PGWP = mapeia pra cada ponto, indices para pontos novos gerados a partir do menos refinado
PWP = 
"""
kl = 1
con9 = 0
LENECI = 0
PNCNF = np.zeros([1,7])
PNECNFN = np.array([[]])
TEN = np.array([])
TMEN = np.array([])
TEEN = np.array([])
MAAN = np.array([[]])
ECI = np.array([[]]) 
NFSNC = 3
NWP = np.zeros([NNG,1])
PGWP = np.zeros([NNG,7])
PWP = np.zeros([NFSNC,7])

"""Subrotina j que procura o elemento da posição [0.5,0.5] e adiciona suas propriedades em NWP, PGWP e PWP
Por ser uma subrotina expecifica do caso L ela não é interessante para generalização"""

for j in range(NNG):
    if COCWB[j][0] == 0.5 and COCWB[j][1] == 0.5:
        NWP[j] = 3
        PGWP[j,0:3] = np.array([1,2,3])
        PWP[0][6] = 1
        PWP[1][1] = 1
        PWP[2][5] = 1
toc()
for i in range(NI):
    tic()
    """Aqui começa a parte de refinamento adaptativo do programa, primeiramente é acriado as matrizes CXNE e CYNE, CXNE é uma matriz 
    onde cada linha representa um elemento do programa sendo que as 3 primeiras posições indicam as 3 posições x dos 3 nós que formam
    aquele elemento e as 3 ultimas posições indicam respectivamente a média entre o x1x2 , x1x3 e x2x3. CYNE trabalha da mesma forma
    CXNE = [x1,x2,x3,(x1+x2)/2,(x1+x3)/2,(x2+x3)/2]"""
    CXNE = np.zeros([NEL,6])
    CYNE = np.zeros([NEL,6])
    for j in range(NEL):
        for k in range(3):
            CXNE[j][k] = np.array(COCWB[int(PNECNF[j][k])][0])
            CYNE[j][k] = np.array(COCWB[int(PNECNF[j][k])][1])
        for k in range(3,6):
            """Note que a posição CXNE de 3 a 6 indica as coordenas do novo triangulo que vai aparecer dentro do triangulo original para
            formar um total de 4 elementos novos a partir de 1 elemento"""
            CXNE[j][k] = (CXNE[j][int(RNE[k][0]-1)] + CXNE[j][int(RNE[k][1]-1)])/2
            CYNE[j][k] = (CYNE[j][int(RNE[k][0]-1)] + CYNE[j][int(RNE[k][1]-1)])/2
    """PNIG = marca quantas base de wavelets já são conhecidas a medida que os nós são escolhidos
    PNIGG = marca o numero do nó que adicionou certa quantidade na mesma posição no PNIG
    NFNG = numero de funções já criadas por wavelets
    NNGI = numero de nós globais a serem refinados
    MHAP2 = marca o indice dos nós que vão ser refinados 
    """
    PNIG = np.zeros([NNG,1])
    PNIGG = np.zeros([NNG,1])
    NFNG = 0
    NNGI = 0
    MHAP2 = np.array([[]])
    mhap2aux = 1
    """Teste em todos os nós para saber quais deles estão presentes em regiões de refinamento, anotando quantos nós vão ser refinados
    além de ficar atento com quantas bases de wavelets já serão criadas por esses nós para fazer o calculo de quantas funções faltam"""
    for j in range(NNG):
        if MHAP[j][0] == 1:
            mhap2aux -= NNGI
            PNIG[j] = np.array(NFNG)
            NFNG += NFN[j]
            NNGI += 1
            MHAP2 = np.append(MHAP2, [[j]], axis = mhap2aux)
            mhap2aux = NNGI
            PNIGG[j] = NNGI
    """
    NFG1 = Numero de funções globais
    NTFG = Numero total de funções globais
    """
    NFG1 = NFNG+NFSNC
    NTFG = NFG1+NTFC
    
    if NFSNC != 0:
        """OCWB_GERADOR_FUNÇÕES_ESPACIAIS
        IGFBGFE é uma matriz com dimeções do numero de nós a serem refinado pelo numero de nós + 3 funções especiais
        PNIGE irá gravar os nós que formam os elementos que receberão as funções globais
        NFE1 é somente o indice 3 que determina que cada elemento é fomado por 3 nós
        NFE2 observa quantas funções terão que ser formada naquele elemento
        NFE3 é somente a soma de NFE1 e NFE2
        IGFBEFE é a solução em cima de cada ponto
        
        """
        GFBEL = np.array([[-1,1,0],[-1,0,1]])
        IGFBGFE = np.zeros([NNGI,NNGI+NFSNC])
        PNIGE = np.zeros([6,1])
        for j in range(NEL):
            NFE1 = 3
            NFE2 = 0
            for k in range(3):
                NFE2 += NWP[int(PNECNF[j][k])]
            NFE3 = NFE1 + NFE2[0]
            """Antes de começar qualquer calculo se faz a solução do elemento que esta sendo observado e se marca nas 3 primeiras 
            posições do PNIGE quais nós foram esse elemento"""
            PNIGE[0:3] = np.array([PNIGG[int(PNECNF[j][0])],PNIGG[int(PNECNF[j][1])],PNIGG[int(PNECNF[j][2])]])
            J  = np.array([[CXNE[j][1]-CXNE[j][0],CYNE[j][1]-CYNE[j][0]],[CXNE[j][2]-CXNE[j][0],CYNE[j][2]-CYNE[j][0]]])
            GFBET = np.linalg.solve(J,GFBEL)      
            IGFBEFE = np.transpose(GFBET).dot(GFBET)*TME[j]*np.abs(np.linalg.det(J))*0.5
            """Agora se testa NFE2 para saber se será necessário fazer um calculo a mais em tais nós, caso não seja necessário
            o processo pula para a geração da matriz de rigidez"""
            if NFE2 > 0:
                """Aqui deveria ser adicionado as novas colunas de IGFBEFE para cada uma das funções que vem de NFE2, infelizmente a
                linha de codigo a seguir não surte efeito no python, para arrumar esse problema foi criado esse if que faz a mesma 
                coisa só que não é genérico
                
                Uma opção que ainda não foi escrita ainda que acredito ser a solução para essa parte é fazer um for
                for k in range(int(NFE2[0])):
                    IGFBEFE = np.append(IGFBEFE, [[0],[0],[0]], axis = 1)
                
                essa linha de codigos no python ai sim teria o mesmo efeito que IGFBEFE[:,int(NFE1):int(NFE3)] = 0 teria no matlab"""
                IGFBEFE[:,int(NFE1):int(NFE3)] = 0
                if len(IGFBEFE[0]) == 3:
                    IGFBEFE = np.append(IGFBEFE, [[0,0,0],[0,0,0],[0,0,0]], axis = 1)
                """PWNE será o peso das wavelets no elemento sendo que o elemento é mapeado da posição 1 a 6 conforme mostrado na 
                definição de RNE
                Nas 3 primeiras linhas e 3 primeiras colunas de PWNE indicam as interações entre os nós originais do elemento interagindo
                com eles mesmo, por isso se coloca 1 em todas essas posições"""
                PWNE = np.zeros([int(NFE3),6])
                PWNE[0:3,0:3] = np.eye(3)
                CONA = 3
                """Agora que a gente tem o elemento que precisa das funções especiais se faz um teste com for para encontrar somente o
                nó que precisa dessas funções"""
                for k in range(3):
                    """k indica qual nó principal do elemento principal esta na condição especial"""
                    for l in range(int(NWP[int(PNECNF[j][k])][0])):
                        """l é um contador para cada uma das 3 funções"""
                        """Encontrado esse nó que no caso L é a dobra se adiciona 3 posições de PNIGE que vão ser os novos 3 nós e para
                        cada um desses nós é adicionado um valor em PWNE, note que agora os numeros adicionados em PWNE são adicionados
                        depois da 3 posição, indicado que são as interações entre os novos elementos criados (CONA > 3 e RNE[k] com k 
                        variando de 0 a 2"""
                        CONA += 1
                        PNIGE[CONA-1] = NNGI+PGWP[int(PNECNF[j][k])][l]
                        for m in range(2):
                            """A primeira linha de PWP indica as interações do elemento 4, a segunda linha do elemento 5 e a terceira 
                            linha do elemento 6. 
                            PPNTE é um vetor com 3 dimenções, a primeira separação é feita para indicar qual dos 3 nós secundários esta
                            sendo observado
                                                                
                            [[[4, 3],  /  referente aos nós 1   
                            [5, 4]],   \                        
                                                                
                            [[2, 7],  / referente aos nós 2
                            [3, 2]],  \
                            
                            [[6, 5],  / referente aos nós 3
                            [7, 6]]]) \
                            
                            3__6__2       3
                            |    /       /|
                            |   /       / |
                           5|  /4     5/  |6
                            | /       /   |
                            |/       /____|
                            1       1  4  2
                            
                            
                            a segunda separação vai indicar, segundo o RNE qual dos 3 nós refinados eu estou me referindo
                            a terceira separação (colunas) indica se estamos nos referindo a um triangulo apontando para baixo (coluna 1)
                            ou um triangulo apontando para cima (coluna 2)
                            
                            A informação que o PPNTE informa é em qual direção que o ponto vai estar de acordo com o anagrama apresentado
                            na variavel TN e NGTN
                            
                            PWP é feito especificamente para esse formato e inica em que direção a quina vai ter interação"""
                            PWNE[CONA-1][int(RNE[k][m]-1)] = np.array(PWP[int(PGWP[int(PNECNF[j][k])][l]-1)][int(PPNTE[k][m][int(TE[j]-1)]-1)])
                GFBE = np.zeros([2,int(NFE3),4])
                for m in range(int(NFE3)):
                    """Gradiente das Funções de Base no Elemento: Ver as funções de base (33-38) página 14 do artigo. Uma linha para cada elemento
                    interno.PWNE - Peso das wavelets no elemento: ver (39) no artigo """
                    GFBE[0:2,m,0] = np.array([-PWNE[m][0]/2+PWNE[m][1]/2+PWNE[m][3],-PWNE[m][0]/2+PWNE[m][2]/2+PWNE[m][4]])
                    GFBE[0:2,m,1] = np.array([-PWNE[m][0]/2+PWNE[m][2]/2-PWNE[m][3]+PWNE[m][5],-PWNE[m][1]/2+PWNE[m][2]/2-PWNE[m][3]+PWNE[m][4]])
                    GFBE[0:2,m,2] = np.array([-PWNE[m][0]/2+PWNE[m][1]/2-PWNE[m][3],-PWNE[m][0]/2+PWNE[m][2]/2-PWNE[m][3]+PWNE[m][5]])
                    GFBE[0:2,m,3] = np.array([-PWNE[m][0]/2+PWNE[m][1]/2-PWNE[m][4]+PWNE[m][5],-PWNE[m][0]/2+PWNE[m][2]/2-PWNE[m][4]])
                for m in range(4):
                    J=np.array([[CXNE[j][REES[m][1][0]-1]-CXNE[j][REES[m][0][0]-1],CYNE[j][REES[m][1][0]-1]-CYNE[j][REES[m][0][0]-1]],[CXNE[j][REES[m][2][0]-1]-CXNE[j][REES[m][0][0]-1],CYNE[j][REES[m][2][0]-1]-CYNE[j][REES[m][0][0]-1]]])
                    Jin = np.linalg.inv(J)
                    GFBET1= Jin.dot(GFBE[:,0:int(NFE1),m])
                    GFBET2= Jin.dot(GFBE[:,int(NFE1):int(NFE3),m])
                    GFBEAUX = np.transpose(GFBET1).dot(GFBET2)*TME[j]*np.abs(np.linalg.det(J))*0.5
                    IGFBEFE[:,int(NFE1):int(NFE3)] += GFBEAUX
            """Junta-se todos os dados de cada nó que sera refinado na malha e forma-se a matriz de rigidez"""
            for k in range(NFE1):
                for l in range(int(NFE3)):
                    IGFBGFE[int(PNIGE[k]-1)][int(PNIGE[l]-1)] += np.array(IGFBEFE[k][l])
        for j in range(NNGI):
            VASU1 = np.array(MHAP2[j][0])
            if int(NNCC[int(VASU1)][0]) == 1:
                IGFBGFE[int(PNIGG[int(VASU1)]-1),:] = 0
                IGFBGFE[:,int(PNIGG[int(VASU1)]-1)] = 0
                IGFBGFE[int(PNIGG[int(VASU1)]-1),int(PNIGG[int(VASU1)]-1)] = 1
        CWN = np.linalg.solve(IGFBGFE[:,0:NNGI],IGFBGFE[:,NNGI:NNGI+NFSNC])
        CWN = -CWN
        CWNL = np.zeros([NNG,NFSNC])
        for k in range(NFSNC):
            for j in range(NNGI):
                VASU1 = np.array(MHAP2[j][0])
                CWNL[int(VASU1)][k] = np.array(CWN[int(PNIGG[int(VASU1)]-1)][k])
    """AQUI TERMINA O PROGRAMA DAS FUNÇÕES ESPACIAIS e começa a parte das condições de contorno de
    dirichlet"""
    IGFBG=np.zeros([NFG1,NTFG])
    IFBEG=np.zeros([NFG1,1])
    for j in range(NEL):
        """
        NFE1 servirá como um contador da soma de quantas funções existentem nos pontos + o numero de funções especiais
        NFCE servirá para indicar o numero total de funções + o numero de nós que possuem valor definido"""
        NFE1 = 0
        NFCE = 0
        for k in range(3):
            NFE1 += NFN[int(PNECNF[j][k])]
            NFCE += int(NNCC[int(PNECNF[j][k])])
        NFE1 += NFSNC
        NFE2=NFE1+NFCE
        IGFBE=np.zeros([NFE1,int(NFE2)])
        IFBE=np.zeros([1,NFE1])
        PNIGE=np.zeros([int(NFE2),1])
        CONA=0
        if NFE1==NFSNC: 
            """isto é, nenhum dos elementos gera uma wavelet, como só as pontas não geram wavelet
            e nenhum elemento é formado por 3 pontas essa parte nunca vai acontecer no programa"""
            GFBELL=np.zeros([2,NFE2])
            FBELL=np.zeros([1,NFE2])
            for k in range(NFSNC):
                CONA+=1
                for l in range(3):
                    GFBELL[:,CONA-1] += np.array(CWNL[int(PNECNF[j][l])][k]*GFBEL[:,l])
                    FBELL[0][CONA-1] += np.array((CWNL[int(PNECNF[j][l])][k])*(FBEL[l]))
                PNIGE[CONA-1] = NFNG + k + 1
            for k in range(3):
                if NNCC[int(PNECNF[j][k])]==1:
                    CONA+=1
                    GFBELL[:,CONA-1]= np.array(GFBEL[:,k])
                    FBELL[0][CONA-1]= np.array(FBEL[k])
                    PNIGE[CONA-1]= np.array(NFG1+PGNCC[int(PNECNF[j][k])])
            J=np.array([[CXNE[j][1]-CXNE[j][0],CYNE[j][1]-CYNE[j][0]],[CXNE[j][2]-CXNE[j][0],CYNE[j,2]-CYNE[j,0]]])
            GFBET1= np.linalg.solve(J,GFBELL[:,0:NFE1])
            GFBET2= np.linalg.solve(J,GFBELL)
            IGFBE= np.transpose(GFBET1).dot(GFBET2)*TME[j]*np.abs(np.linalg.det(J))*0.5
            IFBE= FBELL*TEE[j]*np.abs(np.linalg.det(J))*0.5
        else:
            PWNE=np.zeros([int(NFE2),6])
            for k in range(3):
                """seleciona um dos 3 nós do elemento"""
                for l in range(NFN[int(PNECNF[j][k])]):
                    """rotina determinada pelo numero de funções que são criadas nesse nó, para cada função criada 
                    é adicionado +1 para a variavel CONA e acrescentado no PWNE as interações necessárias entre os 
                    6 nós que formam o elemento"""
                    """interação no nó e depois nos dois nós refinados"""
                    CONA+=1
                    PWNE[CONA-1][k] = np.array(PWTN[int(TN[int(PNECNF[j][k])]-1)][l][0])
                    for m in range(2):
                        PWNE[CONA-1][int(RNE[k][m]-1)] = np.array(PWTN[int(TN[int(PNECNF[j][k])]-1)][l][int(PPNTE[k][m][int(TE[j]-1)]-1)])
                    PNIGE[CONA-1]= PNIG[int(PNECNF[j][k])] + l + 1
            """Salva-se a informação de CONA em CONY"""
            CONY=CONA
            for k in range(NFSNC):
                """acrescenta-se as informações obtidas nas funções especiais"""
                CONA += 1
                if len(PWNE) >= CONA:
                    for l in range(3):
                        PWNE[CONA-1][l] = np.array(CWNL[int(PNECNF[j][l])][k])
                else:
                    PWNE = np.append(PWNE,[[CWNL[int(PNECNF[j][0])][k],CWNL[int(PNECNF[j][1])][k],CWNL[int(PNECNF[j][2])][k],0,0,0]], axis = 0)
                PNIGE[CONA-1] = NFNG + k + 1
            for k in range(3):
                
                for l in range(int(NWP[int(PNECNF[j][k])])):
                    PGXX = np.array(PGWP[int(PNECNF[j][k])][l])
                    for m in range(2):
                        PWNE[int(CONY+PGXX-1)][int(RNE[k][m]-1)]= np.array(PWP[int(PGXX-1)][int(PPNTE[k][m][int(TE[j]-1)]-1)])
            for k in range(3):
                if NNCC[int(PNECNF[j][k])] == 1:                
                    CONA += 1
                    PWNE[CONA-1][k]=1
                    PNIGE[CONA-1]= NFG1 + PGNCC[int(PNECNF[j][k])]
            """Aqui terminamos de montar a PWNE do elemento"""
            GFBE=np.zeros([2,int(NFE2),4])
            FBE=np.zeros([NFE1,4])
            for m in range(int(NFE2)):
                GFBE[0:2,m,0]=np.array([-PWNE[m][0]/2+PWNE[m][1]/2+PWNE[m][3],-PWNE[m][0]/2+PWNE[m][2]/2+PWNE[m][4]])
                GFBE[0:2,m,1]=np.array([-PWNE[m][0]/2+PWNE[m][2]/2-PWNE[m][3]+PWNE[m][5],-PWNE[m][1]/2+PWNE[m][2]/2-PWNE[m][3]+PWNE[m][4]])
                GFBE[0:2,m,2]=np.array([-PWNE[m][0]/2+PWNE[m][1]/2-PWNE[m][3],-PWNE[m][0]/2+PWNE[m][2]/2-PWNE[m][3]+PWNE[m][5]])
                GFBE[0:2,m,3]=np.array([-PWNE[m][0]/2+PWNE[m][1]/2-PWNE[m][4]+PWNE[m][5],-PWNE[m][0]/2+PWNE[m][2]/2-PWNE[m][4]])
                if m + 1 <= NFE1:
                    FBE[m][0]=np.array(PWNE[m][0]*2/3+PWNE[m][1]/6+PWNE[m][2]/6+PWNE[m][3]/3+PWNE[m][4]/3)
                    FBE[m][1]=np.array(PWNE[m][0]/3+PWNE[m][1]/3+PWNE[m][2]/3+PWNE[m][3]/3+PWNE[m][4]/3+PWNE[m][5]/3)
                    FBE[m][2]=np.array(PWNE[m][0]/6+PWNE[m][1]*2/3+PWNE[m][2]/6+PWNE[m][3]/3+PWNE[m][5]/3)
                    FBE[m][3]=np.array(PWNE[m][0]/6+PWNE[m][1]/6+PWNE[m][2]*2/3+PWNE[m][4]/3+PWNE[m][5]/3)
            for m in range(4):
                J= np.array([[CXNE[j][int(REES[m][1][0]-1)]-CXNE[j][int(REES[m][0][0]-1)],CYNE[j][int(REES[m][1][0]-1)]-CYNE[j][int(REES[m][0][0]-1)]],[CXNE[j][int(REES[m][2][0]-1)]-CXNE[j][int(REES[m][0][0]-1)],CYNE[j][int(REES[m][2][0]-1)]-CYNE[j][int(REES[m][0][0]-1)]]])
                GFBET1=np.linalg.solve(J,GFBE[:,0:NFE1,m])
                GFBET2=np.linalg.solve(J,GFBE[:,:,m])
                IGFBE += np.transpose(GFBET1).dot(GFBET2)*TME[j]*np.abs(np.linalg.det(J))*0.5
                FBEaux = FBE[:,m]*TEE[j]*np.abs(np.linalg.det(J))*0.5
                IFBE += FBEaux
        for k in range(NFE1):
            IFBEG[int(PNIGE[k]-1)]+=IFBE[0][k]
            for l in range(int(NFE2)):
                IGFBG[int(PNIGE[k]-1)][int(PNIGE[l]-1)]+=IGFBE[k][l]
    """Aqui ja está pronto a solução de cada elemento"""
    for j in range(NTFC):
        IFBEGaux = SSCC[j]*IGFBG[:,NFG1+j]
        IFBEGaux2 = [[IFBEGaux[0]]]
        for n in range(1,len(IFBEGaux)):
            IFBEGaux2 = np.append(IFBEGaux2,[[IFBEGaux[n]]], axis = 0)
        IFBEG -= IFBEGaux2 
    IGFBG = IGFBG[:,0:NFG1]
    SWI=np.linalg.solve(IGFBG,IFBEG)
#    plt.spy(sps.csr_matrix(IGFBG),marker='.', markersize=2)
#    plt.show()
    toc()
    CON6 = NNG+1
    CONAA=0
    for j in range(NEL):
        if int(MAA[j][0]) == 1:
            for k in range(3,6):
                IND1=0
                for l in range(CON6-1,NNG):
                    if CXNE[j][k]==COCWB[l][0] and CYNE[j][k]==COCWB[l][1]:
                        IND1=1
                        PNECNF[j][k] = l
                if IND1==0:
                    VASU1= np.array(RNE[k,0:2] - 1)
                    VASU2= np.array(PNECNF[j,VASU1[0:2]])
                    VASU3= np.array([int(PPNTE[VASU1[0],0,int(TE[j]-1)]-1),int(PPNTE[VASU1[1],1,int(TE[j]-1)]-1)])
                    VASU4= np.array([int(TN[int(VASU2[0])]-1),int(TN[int(VASU2[1])]-1)])
                    NNG+=1
                    PNECNF[j][k]=NNG-1
                    TN = np.append(TN,[[NGTN[VASU4[0]][VASU3[0]]]], axis = 0)
                    COCWB = np.append(COCWB, [[CXNE[j][k],CYNE[j][k]]], axis = 0)
                    SOCWB = np.append(SOCWB, [0.5*SOCWB[int(VASU2[0])]+0.5*SOCWB[int(VASU2[1])]], axis = 0)
                    for m in range(int(VASU2[0]+1)):
                        if len(PNCNF) <= VASU2[0]:
                            PNCNF = np.append(PNCNF,[[0,0,0,0,0,0,0]], axis = 0)
                        else:
                            PNCNF[int(VASU2[0])][VASU3[0]]= NNG
                            break
                    for m in range(int(VASU2[1]+1)):
                        if len(PNCNF) <= VASU2[1]:
                            PNCNF = np.append(PNCNF,[[0,0,0,0,0,0,0]], axis = 0)
                        else:
                            PNCNF[int(VASU2[1])][VASU3[1]]= NNG
                            break
        elif int(MAA[j][0]) == 2:
            CONAA+=1
            if CONAA == 1:
                MAAL = [[j]]
            else:
                MAAL= np.append(MAAL,[[j]], axis = 0)
    for j in range(CONAA):
        VASU1 = int(MAAL[j]) 
        for k in range(3,6):
            for l in range(CON6-1,NNG):
                if CXNE[VASU1][k]==COCWB[l][0] and CYNE[VASU1][k]==COCWB[l][1]:
                    PNECNF[VASU1][k]=l
                    break
    SWIT= np.zeros([NNG,1])
    for j in range(NNGI):
        VASU1 = int(MHAP2[j][0])
        for k in range(NFN[VASU1]):
            SWIT[VASU1] += PWTN[int(TN[VASU1]-1),k,0]*SWI[int(PNIG[VASU1]+k)]
        for k in range(NFSNC):
            SWIT[VASU1] += CWNL[VASU1][k]*SWI[NFNG+k]
        for k in range(1,7):
            VASU2 = PNCNF[VASU1,k]
            if VASU2 > 0:
                SWIT[int(VASU2-1)] += 0.5*SWIT[VASU1]
                for l in range(NFN[VASU1]):
                    SWIT[int(VASU2-1)] += PWTN[int(TN[VASU1]-1),l,k]*SWI[int(PNIG[VASU1]+l)]
                for l in range(int(NWP[j][0])):
                    SWIT[int(VASU2-1)] += PWP[int(PGWP[VASU1][l]-1)][k]*SWI[int(NFNG+PGWP[VASU1][l]-1)]
    SOCWB += SWIT
    """AQUI terminamos de fazer a solução de um estágio do refinamento e começamos a usar os resultados
    para fazer o refinamento adaptativo """
    if i < NI - 1:
        """CON8 = Contador de elementos
           CON9 = Numero de nós que não irão ser refinados"""
        CON8=0
        CON9=0
        """Primeiramente se adiciona as linhas de MHAP e NFN para cada nó adicionado no ultimo passo"""
        for j in range(CON6-1,NNG):
            MHAP = np.append(MHAP, [[0]], axis = 0)
        for j in range(len(NFN)): #PODE SER QUE ESSA PARTE SEJA DESNECESSÁRIA
            NFN[j] = NFTN[int(TN[j][0]-1)]
        for j in range(len(NFN),len(TN)):
            NFN = np.append(NFN, NFTN[int(TN[j][0]-1)])
        for j in range(NEL):
            """Agora se testa todos os elementos para ver se há a necessidade de refinamento dos mesmos, podendo ser escolhido entre o APOSTERIORI
            e o APRIORI, o APOSTERIORI testa para ver se o erro entre duas soluções esta dentro do linear de erro aceitavel enquanto o APRIORI testa
            se o elemento esta contido dentro da região que deverá ser refinada"""
            VASU1=0
            if APOSTERIORI == 1:
                if int(MAA[j][0]) == 1:
                    for k in range(0,6):
                        if np.abs(SWIT[int(PNECNF[j][k])])>=LDET*np.amax(np.abs(SOCWB)):
                            VASU1=1
                            break
            if APRIORI == 1:
                PEEX=(np.amax(CXNE[j,0:3])+np.amin(CXNE[j,0:3]))/2
                PEEY=(np.amax(CYNE[j,0:3])+np.amin(CYNE[j,0:3]))/2
                if int(MAA[j][0]) == 1:
                    if PEEX >= LX1[i] and PEEX <= LX2[i] and PEEY >= LY1[i] and PEEY <= LY2[i]:
                        VASU1=1
            """VASU1 = 1 representa que o elemento deverá passar por refinamento"""
            if VASU1 == 1:
                for n in range(3,6):
                    MHAP[int(PNECNF[j][n])]= 1
                for n in range(4):
                    if i > 0:
                        if CON8 < NEL:
                            for o in range(3):
                                 PNECNFN[int(CON8),o]= np.array(PNECNF[j,int(REES[n,o,int(TE[j]-1)]-1)])
                            if len(TEN) > CON8:
                                TEN[CON8]= np.array(TEES[n,int(TE[j]-1)])
                            else:
                                TEN = np.append(TEN, [TEES[n,int(TE[j]-1)]])
                            TMEN[CON8]=np.array(TME[j])
                            TEEN[CON8]=np.array(TEE[j])
                            MAAN[CON8] = 1                            
                            CON8+=1
                        else:
                            PNECNFAUX = np.array([PNECNF[j,REES[n,0,int(TE[j]-1)]-1],PNECNF[j,REES[n,1,int(TE[j]-1)]-1],PNECNF[j,REES[n,2,int(TE[j]-1)]-1],0,0,0])
                            PNECNFN = np.append(PNECNFN,[PNECNFAUX], axis = 0)
                            if len(TEN) > CON8:
                                TEN[CON8]= np.array(TEES[n,int(TE[j]-1)])
                            else:
                                TEN = np.append(TEN, [TEES[n,int(TE[j]-1)]])
                            TMEN = np.append(TMEN,[TME[j]])
                            TEEN = np.append(TEEN,[TEE[j]])
                            MAAN = np.append(MAAN,[[1]], axis = 0)
                            CON8 += 1
                    else:
                        kl -= CON8
                        PNECNFAUX = np.array([PNECNF[j,int(REES[n,0,int(TE[j]-1)]-1)],PNECNF[j,int(REES[n,1,int(TE[j]-1)]-1)],PNECNF[j,int(REES[n,2,int(TE[j]-1)]-1)],0,0,0])
                        PNECNFN = np.append(PNECNFN,[PNECNFAUX], axis = kl)
                        TMEN = np.append(TMEN,[TME[j]])
                        TEEN = np.append(TEEN,[TEE[j]])
                        MAAN = np.append(MAAN,[[1]], axis = kl)
                        TEN = np.append(TEN, [TEES[n,int(TE[j]-1)]])
                        CON8 += 1
                        kl = CON8
            else:
                """ECI apresenta os elementos que não precisaram mais apresentar refinamento, no caso ECI"""
                CON9 += 1
                if CON9 == 1 and con9 != 1:
                    ECI = np.append(ECI,[[j]], axis = 1)
                    con9 = 1
                else:
                    if CON9 <= LENECI:
                        ECI[CON9-1] = j
                    else:
                        ECI = np.append(ECI,[[j]], axis = 0)
        if CON9!=NEL:
            for j in range(CON9):
                VASU1 = int(ECI[j][0])
                VASU2 = -8
                CONZX = 0
                if MAA[VASU1][0] > 0:
                    for k in range(3,6):
                        if PNECNF[VASU1][k] > 0:
                            if MHAP[int(PNECNF[VASU1][k])] == 1:
                                VASU2 += k
                                INDXZ = k-3
                                CONZX += 1
                                NFN[int(PNECNF[VASU1][k])] = 0
                for k in range(3):
                    NFN[int(PNECNF[VASU1,k])] = 0
                CONXCA=CON8
                if CON8 < NEL:
                    if CONZX==0:
                        PNECNFN[CON8,0:6] = np.array(PNECNF[VASU1,:])
                        MAAN[CON8] = 0
                        CON8 += 1
                    elif CONZX==1:
                        for k in range(3):
                            PNECNFN[CON8][k]= np.array(PNECNF[VASU1][int(EFN1NC[INDXZ][k]-1)])
                        MAAN[CON8] = 2
                        for k in range(3,6):
                            PNECNFN[CON8][k-3] = np.array(PNECNF[VASU1][int(EFN1NC[INDXZ][k]-1)])
                        MAAN[CON8] = 2
                        CON8+=1
                    elif CONZX==2:
                        for k in range(3):
                            PNECNFN[CON8][k]= np.array(PNECNF[VASU1][int(EFN2NC[VASU2+1][k]-1)])
                        MAAN[CON8] = 2
                        CON8+=1
                        for k in range(3,6):
                            PNECNFN[CON8][k-3]= np.array(PNECNF[VASU1][int(EFN2NC[VASU2+1][k]-1)])
                        MAAN[CON8] = 2
                        CON8+=1
                        for k in range(6,9):
                            PNECNFN[CON8][k-6]= np.array(PNECNF[VASU1][int(EFN2NC[VASU2+1][k]-1)])
                        MAAN[CON8] = 2
                        CON8+=1
                    elif CONZX==3:
                        for k in range(4):
                            MAAN[CON8] = 2
                            for l in range(3):
                                PNECNFN[CON8][l]= np.array(PNECNF[VASU1][int(REES[k,l,int(TE[VASU1]-1)]-1)])
                            CON8+=1
                        for ll in range(CONXCA,CON8):
                            if len(TEN) > ll:
                                TEN[CONXCA:CON8] = np.array(TEES[:,int(TE[VASU1]-1)])
                            else:
                                TEN = np.append(TEN, TEES[:,int(TE[VASU1]-1)])
                    for ll in range(CONXCA,CON8):
                        TEEN[ll] = TEE[VASU1]
                        TMEN[ll] = TME[VASU1]
                else:
                    if CONZX==0:
                        PNECNFN = np.append(PNECNFN,[PNECNF[VASU1,:]], axis = 0)
                        MAAN = np.append(MAAN,[[0]], axis = 0)
                        CON8 += 1
                    elif CONZX==1:
                        PNECNFN = np.append(PNECNFN,[[0,0,0,0,0,0]], axis = 0)
                        for k in range(3):
                            PNECNFN[CON8][k]= np.array(PNECNF[VASU1][int(EFN1NC[INDXZ][k]-1)])
                        MAAN = np.append(MAAN,[[2]], axis = 0)
                        CON8+=1
                        PNECNFN = np.append(PNECNFN,[[0,0,0,0,0,0]], axis = 0)
                        for k in range(3,6):
                            PNECNFN[CON8][k-3] = np.array(PNECNF[VASU1][int(EFN1NC[INDXZ][k]-1)])
                        MAAN = np.append(MAAN,[[2]], axis = 0)
                        CON8+=1
                    elif CONZX==2:
                        PNECNFN = np.append(PNECNFN,[[0,0,0,0,0,0]], axis = 0)
                        for k in range(3):
                            PNECNFN[CON8][k]= np.array(PNECNF[VASU1][int(EFN2NC[VASU2+1][k]-1)])
                        MAAN = np.append(MAAN,[[2]], axis = 0)
                        CON8+=1
                        PNECNFN = np.append(PNECNFN,[[0,0,0,0,0,0]], axis = 0)
                        for k in range(3,6):
                            PNECNFN[CON8][k-3]= np.array(PNECNF[VASU1][int(EFN2NC[VASU2+1][k]-1)])
                        MAAN = np.append(MAAN,[[2]], axis = 0)
                        CON8+=1
                        PNECNFN = np.append(PNECNFN,[[0,0,0,0,0,0]], axis = 0)
                        for k in range(6,9):
                            PNECNFN[CON8][k-6]= np.array(PNECNF[VASU1][int(EFN2NC[VASU2+1][k]-1)])
                        MAAN = np.append(MAAN,[[2]], axis = 0)
                        CON8+=1
                    elif CONZX==3:
                        for k in range(4):
                            CON8+=1
                            MAAN = np.append(MAAN,[[1]], axis = 0)
                            PNECNFN = np.append(PNECNFN,[[0,0,0,0,0,0]], axis = 0)
                            for l in range(3):
                                PNECNFN[CON8-1][l]= np.array(PNECNF[VASU1][int(REES[k,l,int(TE[VASU1]-1)]-1)])
                        for ll in range(CONXCA,CON8):
                            if len(TEN) > ll:
                                TEN[CONXCA:CON8] = np.array(TEES[:,int(TE[VASU1]-1)])
                            else:
                                TEN = np.append(TEN, TEES[:,int(TE[VASU1]-1)])
                    for ll in range(CONXCA,CON8):
                        TEEN = np.append(TEEN, TEE[VASU1])
                        TMEN = np.append(TMEN, TME[VASU1])
        else:
            break
        """Plotação das matrizes auxiliares nas matrizes principais"""
        LENECI = len(ECI)
        NEL=CON8
        PNECNF = np.array(PNECNFN)
        TME = np.append([[]],TMEN)
        TEE = np.append([[]],TEEN)
        MAA = np.array(MAAN)
        TE = np.append([[]],TEN)
        """Adicação de linhas das matrizes de contorno e wavelets primitivas"""
        for k in range(CON6-1,NNG):
            NWP = np.append(NWP,[[0]], axis = 0)
            PGWP = np.append(PGWP, [[0,0,0,0,0,0,0]], axis = 0)
            NNCC = np.append(NNCC, [[0]], axis = 0)
            PGNCC = np.append(PGNCC, [[0]], axis = 0)
        """Teste de cada novo nó para adicionar as propriedades ou não de condição de contorno"""
        for k in range(CON6-1,NNG):
            for l in range(NCC):
                if COCWB[k][0] >= LXCC[l][0] and COCWB[k][0] <= LXCC[l][1] and COCWB[k][1] >= LYCC[l][0] and COCWB[k][1] <= LYCC[l][1]:
                    NTFC+=1
                    NNCC[k] = 1
                    PGNCC[k] = np.array(NTFC)
                    SSCC = np.append(SSCC,[SCC[l]],axis = 0)
                    break

"""Parte de geração do arquivo TXT solução para plotagem"""
arq = open('CasoL.malha', 'w')
arq.write(str(len(COCWB)))
arq.write("\t")
arq.write(str(len(PNECNF)))
arq.write("\t")
arq.write(str(len(SSCC)))
arq.write("\n")
for i in range(len(COCWB)):
    arq.write(str(COCWB[i][0]))
    arq.write("\t")
    arq.write(str(COCWB[i][1]))
    arq.write("\n")
for i in range(len(PNECNF)):
    arq.write(str(int(PNECNF[i][0]+1)))
    arq.write("\t")
    arq.write(str(int(PNECNF[i][1]+1)))
    arq.write("\t")
    arq.write(str(int(PNECNF[i][2]+1)))
    arq.write("\t")
    arq.write("1")
    arq.write("\t")
    arq.write("0.0000000000000000")
    arq.write("\n")
contw = 0
for i in range(len(PGNCC)):
    if PGNCC[i] != 0:
        arq.write(str(int(PGNCC[i][0])))
        arq.write("\t")
        arq.write(str(SSCC[contw]))
        contw += 1
        arq.write("\n")
arq.close()
arq = open('CasoL.resu', 'w')
for i in range(len(SOCWB)):
    arq.write(str(SOCWB[i][0]))
    arq.write("\t")
    arq.write("\n")
arq.close()