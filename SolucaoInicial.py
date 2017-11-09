# -*- coding: utf-8 -*-
import numpy as np
""" 
limtX DETERMINA OS LIMITES DE X PARA CADA DOMINIO;
limtY DETERMINA OS LIMITES DE Y PARA CADA DOMINIO;
nivelRefInicial DETERMINA O NIVEL DE REFINAMENTO INICIAL DE CADA DOMÍNIO;
condContX DETERMINA OS LIMITES DE X PARA AS RETAS DE CONTORNO;
condContY DETERMINA OS LIMITES DE Y PARA AS RETAS DECONTORNO;
condCont DETERMINA O VALOR DO CONTORNO;
matX DETERMINA OS LIMITES DE X PARA CADA MATERIAL;
matY DETERMINA OS LIMITES DE Y PARA CADA MATERIAL;
mat DETERMINA O VALOR DO MATERIAL;
exctX DETERMINA OS LIMITES DE X PARA CADA EXCITAÇÃO;
exctY DETERMINA OS LIMITES DE Y PARA CADA EXCITAÇÃO;
exct DETERMINA O VALOR DA EXCITAÇÃO;
numNos DETERMINA O NÚMERO DE NÓS GLOBAIS;
numEle DETERMINA O NÚMERO DE ELEMENTOS GLOBAIS;
coordNos MATRIZ QUE INDICA AS COORDENADAS DE CADA NÓ;
nosEle MATRIZ QUE INDICA OS NÓS QUE FORAM CERTO ELEMENTO;
coordX VETOR AUXILIAR, ARMAZENA COORDENADAS EM X DE ELEMENTOS;
coordY VETOR AUXILIAR, ARMAZENA COORDENADAS EM Y DE ELEMENTOS;
nosformtri VETOR AUXILIAR, INDICA QUE NÓS FORMAM CADA TRIANGULO EM CERTA SUBROTINA;
posTri MATRIZ QUE INDICA SE O ELEMENTO REFERIDO SÃO DO TIPO 1 /_| OU 2 !¨¨/;
noNumEle VETOR QUE INDICA EM QUANTOS ELEMENTOS TAL NÓ ESTA PRESENTE;
tipoMat VETOR QUE INDICA QUAL O MATERIAL QUE TAL ELEMENTO É COMPOSTO;
tipoExc VETOR QUE INDICA QUAL EXCITAÇÃO TAL ELEMENTO POSSUI;
noContorno VETOR NO QUAL O INDICE INDICA QUAL NO ESTA NO CONTORNO E O VALOR É UM CONTADOR PARA A VARIAVEL valorContorno;
valorContorno VETOR QUE INDICA AS INTENSIDADES DEFINIDAS NOS NÓS DO VETOR noContorno
mRigidez MATRIZ QUE INDICA A INTERAÇÃO ENTRE OS NÓS
mRigEle CONTRIBUIÇÃO DO ELEMENTO COM A MATRIZ DE INTERAÇÕES
vFonte VETOR RESPOSTA
vFontEle CONTRIBUIÇÃO DO ELEMENTO COM O VETOR RESPOSTA 
FunBase FUNÇÃO DE BASE
gradFunBase GRADIENTE DAS FUNÇÕES DE BASE
Jacb MATRIZ JACOBIANA
gradFunBaseTransf GRADIENTE DAS FUNÇÕES DE BASE DEPOIS DE SER TRANSFORMADA PELA MATRIZ JACOBIANA
SI SOLUÇÃO INICIAL
tipoNo INDICA QUAL É O TIPO DO NÓ QUE ESTAMOS TRABALHANDO CONSIDERANDO A REGIÃO EM QUE ELE SE ENCONTRA
tipoNovoNo INDICAR A PARTIR DO DIAGRAMA QUAL SERÁ O TIPO DE NÓ QUANDO ELE FOR GERADO NO REFINAMENTO
numFunWav INDICA QUANTAS FUNÇÕES WAVELETS CADA TIPO DE NÓ GERA
waveltsOrtog MATRIZ QUE ARMAZENA OS COEFICIENTE DE TODAS AS WAVELETS ORTOGONAIS
funWNo VETOR QUE INDICA QUANTAS FUNÇÕES WAVELETS CADA NÓ GERA
"""

# ----------------------------- CARACTERIZAÇÃO DA MALHA -----------------------------

limtX = np.array([[0, 1]])
limtY = np.array([[0, 1]])
nivelRefInicial = np.array([[0.0625, 0.0625]])

condContX = np.array([[0, 1], [0, 1], [0, 0], [1, 1]])
condContY = np.array([[1, 1], [0, 0], [0, 1], [0, 1]])
condCont = np.array([0, 0, 0, 0])

matX = np.array([[0, 1], [0.25, 0.75], [0.25, 0.75]])
matY = np.array([[0, 1], [0.375, 0.4375], [0.5625, 0.625]])
mat = np.array([1, 1000, 1000])

exctX = np.array([[0, 1], [0.25, 0.75], [0.25, 0.75]])
exctY = np.array([[0, 1], [0.375, 0.4375], [0.5625, 0.625]])
exct = np.array([0, -10, 10])

# --------------------------------- GERAÇÃO DA MALHA ---------------------------------

numNos = 0
numEle = 0
coordNos = np.array([[]])
nosEle = np.array([[]])
auxiliarN = 1  # Auxiliar para formação da matriz das Coordenadas dos Nós
auxiliarE = 1  # Auxiliar para formação da matriz de Nós dos Elementos
coordX = np.zeros([3])
coordY = np.zeros([3])

for j in range(len(limtX)):  # len(limtX) dará o numero de domínios
    # Auxiliar que vai moldar as coordenadas X dos triangulos
    moldXtri = [[0, nivelRefInicial[j][0], nivelRefInicial[j][0]], [0, nivelRefInicial[j][0], 0]]
    moldYtri = [[0, 0, nivelRefInicial[j][1]], [0, nivelRefInicial[j][1], nivelRefInicial[j][1]]]
    # Os proximos 2 for's irão calcular quantos elementos vão ser criados no domínio trabalhado
    for k in range(int((limtY[j][1]-limtY[j][0])/nivelRefInicial[j][0])):
        for l in range(int((limtX[j][1]-limtX[j][0])/nivelRefInicial[j][1])):
            for m in range(2):
                nosformtri = np.zeros([3])
                for n in range(3):
                    """As próximas duas linhas modelam nossos elementos retangulares a partir de 2 triangulos.
                    moldXtri = [[0,X,X], e moldYtri = [[0,0,Y], 
                                [0,X,0]]               [0,Y,Y]],
                    Cada uma das 6 posições das matrizes indicam os 6 pontos que formam os 2 triangulos que formam
                    o retangulo de espessura "nivelRefInicial"
                    [0,Y]_______[X,Y]
                        |     /|
                        |    / |
                        |   /  |
                        |  /   |
                        | /    |
                        |/_____|
                    [0,0]      [X,0]"""
                    coordX[n] = limtX[j][0]+l*nivelRefInicial[j][0]+moldXtri[m][n]
                    coordY[n] = limtY[j][0]+k*nivelRefInicial[j][1]+moldYtri[m][n]
                    noExiste = 0
                    for o in range(numNos):
                        if coordX[n] == coordNos[o][0] and coordY[n] == coordNos[o][1]:
                            noExiste = 1
                            nosformtri[n] = o
                            break
                    if noExiste == 0:  # Caso o for não tenha achado o nó com as coord, tal nó é criado.
                        nosformtri[n] = numNos  # Nomeia o número do nó
                        auxiliarN -= numNos
                        # Acrescenta as coordenadas dele
                        coordNos = np.append(coordNos, [[coordX[n], coordY[n]]], axis=auxiliarN)
                        auxiliarN = numNos+1
                        numNos += 1
                # A cada 3 for m é criado um novo elemento (triangulo)
                auxiliarE -= numEle
                nosEle = np.append(nosEle, [nosformtri], axis=auxiliarE)
                auxiliarE = numEle+1
                numEle += 1


# ---------------------------- CARACTERIZAÇÃO DOS ELEMENTOS ----------------------------
"""

posTri 0    posTri 1

2__6__3       2
|    /       /|
|   /       / |
|5 /4      /5 |6
| /       /   |
|/1     1/__4_|3

"""
posTri = np.zeros([numEle])
for j in range(0, numEle, 2):
    posTri[j] = 0
    posTri[j+1] = 1
noNumEle = np.zeros([numNos])
tipoMat = np.zeros([numEle])
tipoExc = np.zeros([numEle])
# busca em que elemento cada nó esta presente
for j in range(numEle):
    for k in range(3):
        noNumEle[int(nosEle[j][k])] += 1
        coordX[k] = coordNos[int(nosEle[j][k])][0]
        coordY[k] = coordNos[int(nosEle[j][k])][1]
    # com as coordenadas dos 3 nós que formam o elemento, calculasse o ponto médio da hipotenusa da triangulo
    pontMedX = (np.amax(coordX)+np.amin(coordX))/2
    pontMedY = (np.amax(coordY)+np.amin(coordY))/2
    indMat = 0  # auxiliar que vai armazenar o indice no qual o elemento se encontra no vetor de materiais
    # procura o material do elemento
    for m in range(len(mat)):
        if matX[m][1] >= pontMedX >= matX[m][0] and matY[m][1] >= pontMedY >= matY[m][0]:
            indMat = m
    tipoMat[j] = mat[indMat]
    indExc = 0  # auxiliar que vai armazenar o indice no qual o elemento se encontra no vetor de excitações
    # procura a excitação do elemento
    for m in range(len(exct)):
        if exctX[m][1] >= pontMedX >= exctX[m][0] and exctY[m][1] >= pontMedY >= exctY[m][0]:
            indExc = m
    tipoExc[j] = exct[indExc]


# ------------------------------- CONDIÇÕES DE CONTORNO -------------------------------

noContorno = np.zeros((numNos, 1))
valorContorno = np.array([])
AuxContorno = 0
# Testa os nós que estão no contorno, salva em noContorno e marca suas intensidades no valorContorno
""" OBS: O valor não é marcado direto em noContorno pois o valor 0 indica que o nó não possui contorno, 
enquanto no vetor valorContorno o 0 indica a intensidade 0 Volts por exemplo"""
for i in range(numNos):
    for j in range(len(condCont)):
        if condContX[j][1] >= coordNos[i][0] >= condContX[j][0] and \
                                condContY[j][1] >= coordNos[i][1] >= condContY[j][0]:
            AuxContorno += 1
            noContorno[i] = AuxContorno 
            valorContorno = np.append(valorContorno, condCont[j])
            break


# ---------------------------------- SOLUÇÃO INICIAL ----------------------------------

mRigidez = np.zeros((numNos, numNos))
vFonte = np.zeros((numNos, 1))
FunBase = np.array([1-1/3-1/3, 1/3, 1/3])  # Para obter uma solução inicial utilizamos funções de bases de lagrange
gradFunBase = np.array([[-1, 1, 0], [-1, 0, 1]])
coordX = np.zeros([numEle, 6])
coordY = np.zeros([numEle, 6])
for i in range(numEle):
    for j in range(3):
        coordX[i][j] = coordNos[int(nosEle[i][j])][0]
        coordY[i][j] = coordNos[int(nosEle[i][j])][1]
    coordX[i][3] = (coordX[i][0]+coordX[i][1])/2
    coordX[i][4] = (coordX[i][0]+coordX[i][2])/2
    coordX[i][5] = (coordX[i][1]+coordX[i][2])/2
    coordY[i][3] = (coordY[i][0]+coordY[i][1])/2
    coordY[i][4] = (coordY[i][0]+coordY[i][2])/2
    coordY[i][5] = (coordY[i][1]+coordY[i][2])/2
    """Agora o vetor coordX e coordY indicam a posição de 6 ponstos do elemento
    2__6__3       2
    |    /       /|
    |   /       / |
    |5 /4      /5 |6
    | /       /   |
    |/1     1/__4_|3
    """
    # Calculo de jacobiana, matriz de rigidez e vetor fonte (contribuição do elemento)
    Jacb = np.array([[coordX[i][1]-coordX[i][0], coordY[i][1]-coordY[i][0]],
                     [coordX[i][2]-coordX[i][0], coordY[i][2]-coordY[i][0]]])
    gradFunBaseTransf = np.linalg.solve(Jacb, gradFunBase)
    # 0.5 vem do calculo da integração no elemento de referencia
    mRigEle = np.transpose(gradFunBaseTransf).dot(gradFunBaseTransf)*tipoMat[i]*np.abs(np.linalg.det(Jacb))*0.5
    vFontEle = FunBase*tipoExc[i]*np.abs(np.linalg.det(Jacb))*0.5
    # jogo minhas contribuições nas matrizes principais
    for j in range(3):
        vFonte[int(nosEle[i][j])] += vFontEle[j]
        for k in range(3):
            mRigidez[int(nosEle[i][j])][int(nosEle[i][k])] += mRigEle[j][k]
# Condições de contorno de Dirichlet
for i in range(numNos):
    if noContorno[i] != 0:
        mRigidez[i] = 0
        mRigidez[i][i] = 1
        vFonte[i] = valorContorno[int(noContorno[i]-1)]
# Solução Inicial
SI = np.linalg.solve(mRigidez, vFonte)


# ----------------------------- BASES WAVELETS ORTOGONAIS -----------------------------

tipoNo = np.ones((numNos, 1))
for j in range(numNos):
    if noNumEle[j] != 6:  # NÓS NO MEIO DA MALHA
        if noNumEle[j] == 4: 
            tipoNo[j] = 10  # QUINA
        elif noNumEle[j] == 3:
            if coordNos[j][0] == 0:
                tipoNo[j] = 4  # NÓS NO EIXO X
            elif coordNos[j][1] == 0:
                tipoNo[j] = 2  # NÓS NO EIXO Y
            elif coordNos[j][1] == limtY[0][1]:
                tipoNo[j] = 3  # NÓS NA DIREITA
            else:
                tipoNo[j] = 5  # NÓS EM CIMA
        elif noNumEle[j] == 2:
            if coordNos[j][0] == 0:
                tipoNo[j] = 6  # nó inferior esquerdo
            else:
                tipoNo[j] = 9  # nó superior direito
        elif coordNos[j][0] == 0:
            tipoNo[j] = 8  # nó inferior direito
        else:
            tipoNo[j] = 7  # nó superior esquerdo
"""
Considere no desenho a seguir o nó 1 sendo o elemento original com tipoNo=2 
tipoNovoNo[1] = [2,2,1,1,2,0,0], isso quer dizer que os pontos 1, 2 e 5 do meu desenho terão tipoNo = 2, 
meus pontos 3 e 4 terão tipoNo = 1 
e meus pontos 6 e 7 terão tipoNo = 0 (não existem)

  ______3_____4
  |    /|    /|
  |   / |   / |
  |  /  |  /  |
  | /   | /   |
 2|/___1|/____|5
  |    /|    /|
  |   / |   / |
  |  /  |  /  |
  | /   | /   |
  |/____|/____|
 7      6
     
     """
tipoN1 = np.array([[-0.3675, 0.2208, 0.3722, 0.4432, 0.5142, 0.3628, 0.2918],
                   [-0.0654, 0.6177, 0.4844, -0.0012, -0.4869, -0.3537, 0.1320],
                   [0.0593, -0.3595, 0.4317, 0.3363, 0.2409, -0.5503, -0.4549]])
numFunWav = np.array([[len(tipoN1)]])
waveletsOrtog = np.zeros([5, len(tipoN1), len(tipoN1[0])])
waveletsOrtog[0, 0:3, 0:7] = tipoN1
tipoNovoNo = np.array([[1, 1, 1, 1, 1, 1, 1]])
tipoN2 = np.array([[0, 0, 0.8944, 0.4472, 0, 0, 0]])
numFunWav = np.append(numFunWav, [[len(tipoN2)]], axis=0)
waveletsOrtog[1, 0:1, 0:7] = tipoN2
tipoNovoNo = np.append(tipoNovoNo, [[2, 2, 1, 1, 2, 0, 0]], axis=0)
tipoN3 = [[0, 0, 0, 0, 0, -0.8944, -0.4472]]
numFunWav = np.append(numFunWav, [[len(tipoN3)]], axis=0)
waveletsOrtog[2, 0:1, 0:7] = tipoN3
tipoNovoNo = np.append(tipoNovoNo, [[3, 3, 0, 0, 3, 1, 1]], axis=0)
tipoN4 = [[0, 0, 0, -0.4472, -0.8944, 0, 0]]
numFunWav = np.append(numFunWav, [[len(tipoN4)]], axis=0)
waveletsOrtog[3, 0:1, 0:7] = tipoN4
tipoNovoNo = np.append(tipoNovoNo, [[4, 0, 4, 1, 1, 4, 0]], axis=0)
tipoN5 = [[0, -0.8944, 0, 0, 0, 0, -0.4472]]
numFunWav = np.append(numFunWav, [[len(tipoN5)]], axis=0)
waveletsOrtog[4, 0:1, 0:7] = tipoN5
tipoNovoNo = np.append(tipoNovoNo, [[5, 1, 5, 0, 0, 5, 1]], axis=0)
numFunWav = np.append(numFunWav, [[0]], axis=0)  # tipo de nós 6,7,8,9 e 10 não geram wavelets
tipoNovoNo = np.append(tipoNovoNo, [[6, 0, 4, 1, 2, 0, 0]], axis=0)
numFunWav = np.append(numFunWav, [[0]], axis=0)
tipoNovoNo = np.append(tipoNovoNo, [[7, 2, 5, 0, 0, 0, 0]], axis=0)
numFunWav = np.append(numFunWav, [[0]], axis=0)
tipoNovoNo = np.append(tipoNovoNo, [[8, 0, 0, 0, 3, 4, 0]], axis=0)
numFunWav = np.append(numFunWav, [[0]], axis=0)
tipoNovoNo = np.append(tipoNovoNo, [[9, 3, 0, 0, 0, 5, 1]], axis=0)
numFunWav = np.append(numFunWav, [[0]], axis=0)
tipoNovoNo = np.append(tipoNovoNo, [[10, 1, 5, 0, 3, 1, 1]], axis=0)

funWNo = np.array(numFunWav[int(tipoNo[0])])
for i in range(1, numNos):
    funWNo = np.append(funWNo, numFunWav[int(tipoNo[i]-1)], axis=0)


# ----------------------------- SAIDAS INICIAIS -----------------------------

SOCWB = SI
COCWB = np.array(coordNos)
PNECNF = np.zeros((numEle, 6))
PNECNF[:, 0:3] = np.array(nosEle)

arq = open('Refin.malha', 'w')
arq.write(str(len(COCWB)))
arq.write("\t")
arq.write(str(len(PNECNF)))
arq.write("\t")
arq.write(str(len(valorContorno)))
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
for i in range(len(noContorno)):
    if noContorno[i] != 0:
        arq.write(str(int(noContorno[i][0])))
        arq.write("\t")
        arq.write(str(valorContorno[contw]))
        contw += 1
        arq.write("\n")
arq.close()
arq = open('Refin.resu', 'w')
for i in range(len(SI)):
    arq.write(str(SI[i][0]))
    arq.write("\t")
    arq.write("\n")
arq.close()
