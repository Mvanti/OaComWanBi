# -*- coding: utf-8 -*-
import SolucaoInicial
import SalvaMalha
import numpy as np

"""
numInteracoes INDICA QUANTAS ITERAÇÕES SERÃO PROCESSADAS ALÉM DA SOLUÇÃO INICIAL;
nivelErro INDICA QUAL SERÁ O NIVEL DE ERRO ACEITO ENTRE DUAS ITERAÇÕES;
posNos MATRIZ FINAL DO POSICIONAMENTO DE CADA UM DOS NÓS;
ampNos MATRIZ SOLUÇÃO DO PROBLEMA EM CIMA DE CADA UM DOS NÓS;
nosEle MATRIZ QUE INDICA QUAIS NÓS FORMAM CADA UM DOS ELEMENTOS;
nosRefin VETOR QUE INDICA QUAIS NÓS GLOBAIS VÃO PASSAR POR REFINAMENTO; (1 Sim / 0 Não)
eleRefin VETOR QUE INDICA QUAIS ELEMENTOS VÃO PASSAR POR REFINAMENTO; (1 Sim / 0 Não)
coordX VETOR AUXILIAR, ARMAZENA COORDENADAS EM X DOS NÓS QUE FORMAM OS ELEMENTOS QUE ESTÃO SENDO PROCESSADOS;
coordY VETOR AUXILIAR, ARMAZENA COORDENADAS EM Y DOS NÓS QUE FORMAM OS ELEMENTOS QUE ESTÃO SENDO PROCESSADOS;
indNosRefin VETOR QUE GRAVA O INDICE DOS NÓS QUE SERÃO REFINADOS;
numFunc INDICA QUANTAS BASES WAVELETS SERÃO CRIADAS A PARTIR DOS NÓS QUE SERÃO PROCESSADOS;
mRigidez MATRIZ QUE INDICA A INTERAÇÃO ENTRE OS NÓS;
vFonte VETOR RESPOSTA;
funcNo INDICA QUANTAS BASES WAVELETS SERÃO CRIADAS A PARTIR DOS NÓS QUE ESTÃO EM UM CERTO ELEMENTO;
noCont INDICA QUANTOS NÓS ESTÃO NO CONTORNOR EM UM CERTO ELEMENTO;
mRigEle CONTRIBUIÇÃO DO ELEMENTO COM A MATRIZ DE INTERAÇÕES;
vFontEle CONTRIBUIÇÃO DO ELEMENTO COM O VETOR RESPOSTA;
Ele INDICA EM QUE POSIÇÕES DAS MATRIZES GLOBAIS VÃO SER POSICIONADAS AS SOLUÇÕES DO ELEMENTO;  
pWvNoEl É A MATRIZ QUE CALCULA O PESO DAS FUNÇÕES WAVELETS PROVENIENTE DE UM NÓS SOBRE OS NÓS VISINHOS;
gradFunBaseEle GRADIENTE DAS FUNÇÕES DE BASE DOS ELEMENTOS;
FunBaseEle FUNÇÕES DE BASE DO ELEMENTO;
J MATRIZ JACOBIANA PARA TRANSFORMAÇÃO DE BASE;
nosVizinhos INDICA QUAIS NÓS SÃO VIZINHOS CONSIDERANDO OS ELEMENTOS DE REFERENCIA;
indNosVizin INDICA O INDICE DOS NÓS VIZINHOS;
pRefCriados INDICA A LOCALIZAÇÃO DO PONTO CRIADO EM RELAÇÃO AOS PONTOS VIZINHOS DE REFERENCIA;
tipoNosVizin INDICA QUAL É O TIPO DE NO DOS NÓS VIZINHOS (consulte tipoNo);
eleRefinL INDICA ELEMENTOS  QUE ESTÃO NA QUINA DO "L";
Erro DIFERENÇA ENTRE SOLUÇÃO V1 E V2
"""

# ---------------------------------- Carregamento de dados de SolucaoInicial.py ----------------------------------------
posNos = SolucaoInicial.COCWB
ampNos = SolucaoInicial.SOCWB
nosEle = SolucaoInicial.PNECNF
numEle = SolucaoInicial.numEle
numNos = SolucaoInicial.numNos
tipoMat = SolucaoInicial.tipoMat
tipoExc = SolucaoInicial.tipoExc
tipoNo = SolucaoInicial.tipoNo
tipoNovoNo = SolucaoInicial.tipoNovoNo
posTri = SolucaoInicial.posTri
funWNo = SolucaoInicial.funWNo
condCont = SolucaoInicial.condCont
condContX = SolucaoInicial.condContX
condContY = SolucaoInicial.condContY
AuxContorno = SolucaoInicial.AuxContorno
noContorno = SolucaoInicial.noContorno
valorContorno = SolucaoInicial.valorContorno
waveletsOrtog = SolucaoInicial.waveletsOrtog
numFunWav = SolucaoInicial.numFunWav

# -------------------------------------------- MATRIZES AUXILIARES -----------------------------------------------------
mapNos = np.array([[4, 3], [3, 5], [5, 4], [1, 0], [0, 2], [2, 1]])
"""
mapNos servirá para fazer o mapeamento dos nós do triangulo e suas respectivas localizações
 2
 |\  
 | \ 
4|__\ 5
 |\  |\ 
 | \ | \ 
 |__\|__\ 
 0   3   1

mapNos[0] = linha entre 4 e 3 {
mapNos[1] = linha entre 3 e 5 { Triangulo Refinado
mapNos[2] = linha entre 5 e 4 {
mapNos[3] = linha entre 1 e 0 {
mapNos[4] = linha entre 0 e 2 { Triangulo Original
mapNos[5] = linha entre 2 e 1 {
"""
pNoRefin = np.array([[[3, 2],
                      [4, 3]],
                     [[1, 6],
                      [2, 1]],
                     [[5, 4],
                      [6, 5]]])
"""
pNoRefin servirá para auxiliar a encontrar os pontos de refinamento ao redor do nó de referência, perceba que 
ele leva em conta em qual posição do triangulo o nó de referencia se encontra, o tipo de triangulo que ele possui
                            
  TIPO 0    TIPO 1

 2__6__3       2
 |    /       /|
 |   /       / |
 |5 /4      /5 |6
 | /       /   |
 |/1     1/__4_|3

Tipo 0    1
     ||   ||
     \/   \/
Se o nó que estamos trabalhando for o nó 1 no elemento de referência
  
  [[[3,  2],  
    [4,  3]],                      

Se o nó que estamos trabalhando for o nó 2 no elemento de referência                               

   [[1,  6],  
    [2,  1]],  

Se o nó que estamos trabalhando for o nó 3 no elemento de referência

   [[5,  4],  
    [6,  5]]]) 

Note que 
  _____________
  |    /|    /|
  |   / |   / |
  |  /  |4 /3 |
  | /   | /   |
  |/_5__|0__2_|
  |    /|    /|
  |   / |   / |
  |  /6 |1 /  |
  | /   | /   |
  |/____|/____|
  

"""
refNosEle = np.array([[[0, 0], [3, 3], [4, 4]],
                      [[3, 4], [5, 3], [4, 5]],
                      [[3, 3], [1, 1], [5, 5]],
                      [[4, 4], [5, 5], [2, 2]]])
"""
refNosEle é uma matriz que indica quais nós do ELEMENTO DE REFERENCIA que formaração cada um dos 4 novos elementos
formados a partir do elemento que foi refinado

refNosEle = np.array([[[0, 0],  }
                       [3, 3],  } PRIMEIRO ELEMENTO
                       [4, 4]], }

                      [[3, 4],  }
                       [5, 3],  } SEGUNDO ELEMENTO
                       [4, 5]], } 

                      [[3, 3],  }
                       [1, 1],  } TERCEIRO ELEMENTO
                       [5, 5]], }

                      [[4, 4],  }
                       [5, 5],  } QUATO ELEMENTO
                       [2, 2]]])}
 2
 |\ 
 | \ 
 |4 \ 
4|___\ 5
 |\ 2|\ 
 |1\ | \ 
 |__\|3_\ 
 0   3   1
"""
posTriRefin = np.array([[0, 1], [1, 0], [0, 1], [0, 1]])
"""
posTriRefin organiza qual será a posição dos novos elementos criados a partir de outro elemento

  TIPO 0    TIPO 1

 2__6__3       2
 | /| /       /|
 |/ |/       /_|
5|__/4     5/  |6
 | /       /| /|
 |/1     1/_|/_|3
            4
[[ 0    ,    1], tipo do elemento refinado 1 
 [ 1    ,    0], tipo do elemento refinado 2
 [ 0    ,    1], tipo do elemento refinado 3
 [ 0    ,    1]] tipo do elemento refinado 4
 
perceba que o elemento do tipo 0 refinado gera três elementos tipo 0 e um elemento tipo 1
o elemento tipo 1 refinado gera três elementos tipo 1 e um elemento tipo 0
 
"""
"""

a varAux3 serve para momentos em que existem 2 nós prontos

 __6__       
|    /       /|
|   /       / |
|5 /4      /5 |6
| /       /   |
|/       /__4_|

Se o 4 e o 5 estiverem prontos, varAux3 + 3 + 4 = 0
Se o 4 e o 6 estiverem prontos, varAux3 + 3 + 5 = 1
Se o 5 e o 6 estiverem prontos, varAux3 + 4 + 5 = 2

a varAux4 serve para momentos em que existe 1 nó pronto

Se o 4 estiver pronto, varAux4 = 0
Se o 5 estiver pronto, varAux4 = 1
Se o 6 estiver pronto, varAux4 = 2
"""
EFN2NC = np.array([[0, 3, 4, 4, 3, 1, 4, 1, 2], [0, 5, 2, 0, 3, 5, 3, 1, 5], [0, 5, 4, 0, 1, 5, 4, 5, 2]])
EFN1NC = np.array([[0, 3, 2, 3, 1, 2], [0, 1, 4, 4, 1, 2], [0, 1, 5, 0, 5, 2]])

nosEleAux = np.zeros([1, 7])
eleRefinL = np.array([[]])
posTriNovo = np.array([[0]])
tipoMatNovo = np.array([[0]])
tipoExcNovo = np.array([[0]])
eleRefinNovo = np.array([[0]])
novoNosEle = np.array([[0, 0, 0, 0, 0, 0]])
indElePront = np.array([[0]])
varAux2 = 1
lenIndElePront = 0

# ---------------------------------------- CONDIÇÕES PARA PRIMEIRA ITERAÇÃO --------------------------------------------
numIteracoes = 3
APOSTERIORI = 0
nivelErro = 0.0025
APRIORI = 1
LX1 = np.array([0.25, 0.25, 0.25, 0.25])
LX2 = np.array([0.75, 0.75, 0.75, 0.75])
LY1 = np.array([0.375, 0.375, 0.375, 0.375])
LY2 = np.array([0.625, 0.625, 0.625, 0.625])
nosRefin = np.ones((numNos, 1))
eleRefin = np.ones((numEle, 1))
# ----------------------------------------------- LOOPING DE ITERAÇÕES -------------------------------------------------

for i in range(int(numIteracoes)):
    # ---------------------------------- POSICIONAMENTO DOS NÓS ------------------------------------------------
    coordX = np.zeros([numEle, 6])
    coordY = np.zeros([numEle, 6])
    for j in range(numEle):
        for k in range(3):
            coordX[j][k] = np.array(posNos[int(nosEle[j][k])][0])
            coordY[j][k] = np.array(posNos[int(nosEle[j][k])][1])
        for k in range(3, 6):
            coordX[j][k] = (coordX[j][int(mapNos[k][0])] + coordX[j][int(mapNos[k][1])])/2
            coordY[j][k] = (coordY[j][int(mapNos[k][0])] + coordY[j][int(mapNos[k][1])])/2

    # --------------------------- DADOS SOBRE OS NÓS QUE SERÃO REFINADOS ---------------------------------------
    """  
    quantBaseWav irá anotar quantas Funções de base já existem depois de N nós processados
    nosBaseWav indica quantos nós irão formar funções de Base Wavelets
    """
    quantBaseWav = np.zeros([numNos, 1])
    nosBaseWav = np.zeros([numNos, 1])
    indNosRefin = np.array([[]], dtype=int)
    numNosRefin = 0
    numFunc = 0
    refinAux = 1
    for j in range(numNos):
        if nosRefin[j][0] == 1:
            refinAux -= numNosRefin
            quantBaseWav[j] = np.array(numFunc)
            numFunc += funWNo[j]
            numNosRefin += 1
            indNosRefin = np.append(indNosRefin, [[j]], axis=refinAux)
            refinAux = numNosRefin
            nosBaseWav[j] = numNosRefin

    # ---------------------------- SOLUÇÃO DA NOVA MALHA DE REFINAMENT------------------------------------------

    mRigidez = np.zeros([numFunc, numFunc + len(valorContorno)])
    vFonte = np.zeros([numFunc, 1])

    for j in range(numEle):
        # ----------------------------- SOLUÇÃO DE CADA ELEMENTO -----------------------------------------------
        funcNo = 0
        noCont = 0
        for k in range(3):
            funcNo += funWNo[int(nosEle[j][k])]
            if int(noContorno[int(nosEle[j][k])]) != 0:
                noCont += 1
        mRigEle = np.zeros([funcNo, int(funcNo+noCont)])
        vFontEle = np.zeros([1, funcNo])
        pSoluEle = np.zeros([int(funcNo+noCont), 1])
        contAux = 0
        pWvNoEl = np.zeros([int(funcNo+noCont), 6])
        for k in range(3):
            for l in range(funWNo[int(nosEle[j][k])]):
                pWvNoEl[contAux][k] = np.array(waveletsOrtog[int(tipoNo[int(nosEle[j][k])]-1)][l][0])
                for m in range(2):
                    pWvNoEl[contAux][int(mapNos[k][m])] = \
                        waveletsOrtog[int(tipoNo[int(nosEle[j][k])]-1)][l][int(pNoRefin[k][m][int(posTri[j])])]
                pSoluEle[contAux] = quantBaseWav[int(nosEle[j][k])] + l + 1
                contAux += 1
        for k in range(3):
            if noContorno[int(nosEle[j][k])] != 0:
                pWvNoEl[contAux][k] = 1
                pSoluEle[contAux] = numFunc + noContorno[int(nosEle[j][k])]
                contAux += 1
        gradFunBaseEle = np.zeros([2, int(funcNo+noCont), 4])
        FunBaseEle = np.zeros([funcNo, 4])
        for m in range(int(funcNo+noCont)):
            gradFunBaseEle[0:2, m, 0] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][1]/2+pWvNoEl[m][3],
                                                  -pWvNoEl[m][0]/2+pWvNoEl[m][2]/2+pWvNoEl[m][4]])

            gradFunBaseEle[0:2, m, 1] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][2]/2-pWvNoEl[m][3]+pWvNoEl[m][5],
                                                  -pWvNoEl[m][1]/2+pWvNoEl[m][2]/2-pWvNoEl[m][3]+pWvNoEl[m][4]])

            gradFunBaseEle[0:2, m, 2] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][1]/2-pWvNoEl[m][3],
                                                  -pWvNoEl[m][0]/2+pWvNoEl[m][2]/2-pWvNoEl[m][3]+pWvNoEl[m][5]])

            gradFunBaseEle[0:2, m, 3] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][1]/2-pWvNoEl[m][4]+pWvNoEl[m][5],
                                                  -pWvNoEl[m][0]/2+pWvNoEl[m][2]/2-pWvNoEl[m][4]])
            if m + 1 <= funcNo:
                FunBaseEle[m][0] = np.array(pWvNoEl[m][0]*2/3+pWvNoEl[m][1]/6+pWvNoEl[m][2]/6 +
                                            pWvNoEl[m][3]/3+pWvNoEl[m][4]/3)

                FunBaseEle[m][1] = np.array(pWvNoEl[m][0]/3+pWvNoEl[m][1]/3+pWvNoEl[m][2]/3+pWvNoEl[m][3]/3 +
                                            pWvNoEl[m][4]/3+pWvNoEl[m][5]/3)

                FunBaseEle[m][2] = np.array(pWvNoEl[m][0]/6+pWvNoEl[m][1]*2/3+pWvNoEl[m][2]/6 +
                                            pWvNoEl[m][3]/3+pWvNoEl[m][5]/3)

                FunBaseEle[m][3] = np.array(pWvNoEl[m][0]/6+pWvNoEl[m][1]/6+pWvNoEl[m][2]*2/3 +
                                            pWvNoEl[m][4]/3+pWvNoEl[m][5]/3)
        for m in range(4):
            J = np.array([[coordX[j][int(refNosEle[m][1][0])]-coordX[j][int(refNosEle[m][0][0])],
                           coordY[j][int(refNosEle[m][1][0])]-coordY[j][int(refNosEle[m][0][0])]],
                          [coordX[j][int(refNosEle[m][2][0])]-coordX[j][int(refNosEle[m][0][0])],
                           coordY[j][int(refNosEle[m][2][0])]-coordY[j][int(refNosEle[m][0][0])]]])
            gradFunBaseEleTransf1 = np.linalg.solve(J, gradFunBaseEle[:, 0:funcNo, m])
            gradFunBaseEleTransf2 = np.linalg.solve(J, gradFunBaseEle[:, :, m])
            mRigEle += np.transpose(gradFunBaseEleTransf1).dot(
                    gradFunBaseEleTransf2)*tipoMat[j]*np.abs(np.linalg.det(J))*0.5
            FunBaseEleAux = FunBaseEle[:, m]*tipoExc[j]*np.abs(np.linalg.det(J))*0.5
            vFontEle += FunBaseEleAux
        for k in range(funcNo):
            vFonte[int(pSoluEle[k]-1)] += vFontEle[0][k]
            for l in range(int(funcNo+noCont)):
                mRigidez[int(pSoluEle[k]-1)][int(pSoluEle[l]-1)] += mRigEle[k][l]
    for j in range(len(valorContorno)):
        vFonte -= np.transpose([valorContorno[j]*mRigidez[:, numFunc+j]])
    mRigidez = mRigidez[:, 0:numFunc]
    Solve = np.linalg.solve(mRigidez, vFonte)

    # ---------------------------------- CRIAÇÃO DOS NOVOS NÓS -------------------------------------------------
    antNumNos = numNos
    contAux2 = 0
    for j in range(numEle):
        if int(eleRefin[j][0]) == 1:
            for k in range(3, 6):
                noExiste = 0
                for l in range(antNumNos, numNos):
                    if coordX[j][k] == posNos[l][0] and coordY[j][k] == posNos[l][1]:
                        noExiste = 1
                        nosEle[j][k] = l
                if noExiste == 0:
                    nosVizinhos = np.array(mapNos[k, 0:2])
                    indNosVizin = np.array(nosEle[j, nosVizinhos[0:2]])
                    pRefCriados = np.array([int(pNoRefin[nosVizinhos[0], 0, int(posTri[j])]),
                                            int(pNoRefin[nosVizinhos[1], 1, int(posTri[j])])])
                    tipoNosVizin = np.array([int(tipoNo[int(indNosVizin[0])]-1),
                                             int(tipoNo[int(indNosVizin[1])]-1)])
                    nosEle[j][k] = numNos
                    numNos += 1
                    tipoNo = np.append(tipoNo, [[tipoNovoNo[tipoNosVizin[0]][pRefCriados[0]]]], axis=0)
                    posNos = np.append(posNos, [[coordX[j][k], coordY[j][k]]], axis=0)
                    ampNos = np.append(ampNos, [0.5*ampNos[int(indNosVizin[0])] +
                                                0.5*ampNos[int(indNosVizin[1])]], axis=0)

                    while len(nosEleAux) <= indNosVizin[0] or len(nosEleAux) <= indNosVizin[1]:
                        nosEleAux = np.append(nosEleAux, [[0, 0, 0, 0, 0, 0, 0]], axis=0)
                    nosEleAux[int(indNosVizin[0])][pRefCriados[0]] = numNos
                    nosEleAux[int(indNosVizin[1])][pRefCriados[1]] = numNos

    # ----------------------------------- CALCULO E ADIÇÃO DO ERRO A SOLUÇÃO -----------------------------------
    Erro = np.zeros([numNos, 1])
    for j in range(numNosRefin):
        for k in range(funWNo[int(indNosRefin[j][0])]):
            Erro[int(indNosRefin[j][0])] += waveletsOrtog[int(tipoNo[int(indNosRefin[j][0])]-1), k, 0] * \
                                            Solve[int(quantBaseWav[int(indNosRefin[j][0])]+k)]
        for k in range(1, 7):
            nosEleAux_ = nosEleAux[int(indNosRefin[j][0]), k]
            if nosEleAux_ > 0:
                Erro[int(nosEleAux_-1)] += 0.5 * Erro[int(indNosRefin[j][0])]
                for l in range(funWNo[int(indNosRefin[j][0])]):
                    Erro[int(nosEleAux_-1)] += waveletsOrtog[int(tipoNo[int(indNosRefin[j][0])]-1), l, k] * \
                                          Solve[int(quantBaseWav[int(indNosRefin[j][0])]+l)]
    ampNos += Erro

    # ------------------------------------ RECONSTRUÇÃO DA MALHA DE TRABALHO -----------------------------------
    if i < numIteracoes - 1:
        contEle = 0
        numNosNRefin = 0
        for j in range(antNumNos, numNos):
            nosRefin = np.append(nosRefin, [[0]], axis=0)
        for j in range(len(funWNo)):
            funWNo[j] = numFunWav[int(tipoNo[j][0]-1)]
        for j in range(len(funWNo), len(tipoNo)):
            funWNo = np.append(funWNo, numFunWav[int(tipoNo[j][0]-1)])
        for j in range(numEle):
            testeRefin = 0
            if APOSTERIORI == 1:
                if int(eleRefin[j][0]) == 1:
                    for k in range(0, 6):
                        if np.abs(Erro[int(nosEle[j][k])]) >= nivelErro*np.amax(np.abs(ampNos)):
                            testeRefin = 1
                            break
            if APRIORI == 1:
                PEEX = (np.amax(coordX[j, 0:3])+np.amin(coordX[j, 0:3]))/2
                PEEY = (np.amax(coordY[j, 0:3])+np.amin(coordY[j, 0:3]))/2
                if int(eleRefin[j][0]) == 1:
                    if PEEX >= LX1[i] and PEEX <= LX2[i] and PEEY >= LY1[i] and PEEY <= LY2[i]:
                        testeRefin = 1
            if testeRefin == 1:
                for n in range(3, 6):
                    nosRefin[int(nosEle[j][n])] = 1
                for n in range(4):
                    while len(novoNosEle) <= contEle:
                        novoNosEle = np.append(novoNosEle, [[0, 0, 0, 0, 0, 0]], axis=0)
                    while len(posTriNovo) <= contEle:
                        posTriNovo = np.append(posTriNovo, [[0]], axis=0)
                    while len(tipoMatNovo) <= contEle:
                        tipoMatNovo = np.append(tipoMatNovo, [[0]], axis=0)
                    while len(tipoExcNovo) <= contEle:
                        tipoExcNovo = np.append(tipoExcNovo, [[0]], axis=0)
                    while len(eleRefinNovo) <= contEle:
                        eleRefinNovo = np.append(eleRefinNovo, [[0]], axis=0)
                    for o in range(3):
                        novoNosEle[int(contEle)][o] = int(nosEle[j][int(refNosEle[n, o, int(posTri[j])])])
                    posTriNovo[contEle] = np.array(posTriRefin[n, int(posTri[j])])
                    tipoMatNovo[contEle] = np.array(tipoMat[j])
                    tipoExcNovo[contEle] = np.array(tipoExc[j])
                    eleRefinNovo[contEle] = 1
                    contEle += 1
            else:
                while len(indElePront) <= numNosNRefin:
                    indElePront = np.append(indElePront, [[0]], axis=0)
                indElePront[numNosNRefin] = j
                numNosNRefin += 1
        # ------------------------------------ ELEMENTOS PRONTOS -------------------------------------------------------
        if numNosNRefin != numEle:
            for j in range(numNosNRefin):
                indEleP = int(indElePront[j][0])
                varAux3 = -7
                numNosPront = 0
                if eleRefin[indEleP][0] > 0:
                    for k in range(3, 6):
                        if nosEle[indEleP][k] > 0:
                            if nosRefin[int(nosEle[indEleP][k])] == 1:
                                varAux3 += k
                                varAux4 = k-3
                                numNosPront += 1
                                funWNo[int(nosEle[indEleP][k])] = 0
                for k in range(3):
                    funWNo[int(nosEle[indEleP, k])] = 0
                antNumEle = contEle
                while len(novoNosEle) <= contEle:
                    novoNosEle = np.append(novoNosEle, [[0, 0, 0, 0, 0, 0]], axis=0)
                while len(eleRefinNovo) <= contEle:
                    eleRefinNovo = np.append(eleRefinNovo, [[0]], axis=0)
                if numNosPront == 0:
                    novoNosEle[contEle, 0:6] = np.array(nosEle[indEleP, :])
                    eleRefinNovo[contEle] = 0
                    contEle += 1
                elif numNosPront == 1:
                    for k in range(3):
                        novoNosEle[contEle][k] = np.array(nosEle[indEleP][int(EFN1NC[varAux4][k])])
                    eleRefinNovo[contEle] = 2
                    contEle += 1
                    while len(novoNosEle) <= contEle:
                        novoNosEle = np.append(novoNosEle, [[0, 0, 0, 0, 0, 0]], axis=0)
                    while len(eleRefinNovo) <= contEle:
                        eleRefinNovo = np.append(eleRefinNovo, [[0]], axis=0)
                    for k in range(3, 6):
                        novoNosEle[contEle][k-3] = np.array(nosEle[indEleP][int(EFN1NC[varAux4][k])])
                    eleRefinNovo[contEle] = 2
                    contEle += 1
                elif numNosPront == 2:
                    for k in range(3):
                        novoNosEle[contEle][k] = np.array(nosEle[indEleP][int(EFN2NC[varAux3][k])])
                    eleRefinNovo[contEle] = 2
                    contEle += 1
                    while len(novoNosEle) <= contEle:
                        novoNosEle = np.append(novoNosEle, [[0, 0, 0, 0, 0, 0]], axis=0)
                    while len(eleRefinNovo) <= contEle:
                        eleRefinNovo = np.append(eleRefinNovo, [[0]], axis=0)
                    for k in range(3, 6):
                        novoNosEle[contEle][k-3] = np.array(nosEle[indEleP][int(EFN2NC[varAux3][k])])
                    eleRefinNovo[contEle] = 2
                    contEle += 1
                    while len(novoNosEle) <= contEle:
                        novoNosEle = np.append(novoNosEle, [[0, 0, 0, 0, 0, 0]], axis=0)
                    while len(eleRefinNovo) <= contEle:
                        eleRefinNovo = np.append(eleRefinNovo, [[0]], axis=0)
                    for k in range(6, 9):
                        novoNosEle[contEle][k-6] = np.array(nosEle[indEleP][int(EFN2NC[varAux3][k])])
                    eleRefinNovo[contEle] = 2
                    contEle += 1
                elif numNosPront == 3:
                    for k in range(4):
                        while len(novoNosEle) <= contEle:
                            novoNosEle = np.append(novoNosEle, [[0, 0, 0, 0, 0, 0]], axis=0)
                        while len(eleRefinNovo) <= contEle:
                            eleRefinNovo = np.append(eleRefinNovo, [[0]], axis=0)
                        eleRefinNovo[contEle] = 1
                        for l in range(3):
                            novoNosEle[contEle][l] = np.array(
                                nosEle[indEleP][int(refNosEle[k, l, int(posTri[indEleP])])])
                        contEle += 1
                    while len(posTriNovo) <= contEle:
                        posTriNovo = np.append(posTriNovo, [[0]], axis=0)
                    for m in range(4):
                        posTriNovo[contEle-3+m] = np.array(posTriRefin[m, int(posTri[indEleP])])
                while len(tipoExcNovo) <= contEle:
                    tipoExcNovo = np.append(tipoExcNovo, [[0]], axis=0)
                while len(tipoMatNovo) <= contEle:
                    tipoMatNovo = np.append(tipoMatNovo, [[0]], axis=0)
                for ll in range(antNumEle, contEle):
                    tipoExcNovo[ll] = tipoExc[indEleP]
                    tipoMatNovo[ll] = tipoMat[indEleP]
        else:
            break
        # -------------------------------------------- ARRUMANDO MATRIZES ----------------------------------------------
        lenIndElePront = len(indElePront)
        numEle = contEle
        nosEle = np.array(novoNosEle)
        tipoMat = np.append([[]], tipoMatNovo)
        tipoExc = np.append([[]], tipoExcNovo)
        eleRefin = np.array(eleRefinNovo)
        posTri = np.array(posTriNovo)
        for k in range(antNumNos, numNos):
            noContorno = np.append(noContorno, [[0]], axis=0)
        for k in range(antNumNos, numNos):
            for l in range(len(condCont)):
                if condContX[l][1] >= posNos[k][0] >= condContX[l][0] and \
                                        condContY[l][1] >= posNos[k][1] >= condContY[l][0]:
                    AuxContorno += 1
                    noContorno[k] = np.array(AuxContorno)
                    valorContorno = np.append(valorContorno, [condCont[l]], axis=0)
                    break
SalvaMalha.SM(posNos, ampNos, nosEle, valorContorno, noContorno)
