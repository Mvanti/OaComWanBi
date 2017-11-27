import numpy as np
NFE = np.zeros(2)
posNos = np.array([[0, 0], [0.5, 0], [1, 0], [0, 0.5], [0.5, 0.5], 
                   [1, 0.5], [0, 1], [0.5, 1], [1, 1]])
nosEle = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], 
                   [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]])
noRef = 4
NHCW = np.array([0, 1, 2, 3, 4, 5, 6])
NFBS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
matX = np.array([[0, 1]])
matY = np.array([[0, 1]])
mat = np.array([1])

numNos = len(posNos)
numEle = len(nosEle)
numMat = len(mat)
mat = np.append(mat, [1], axis=0)
posTri = np.array([0, 1, 0, 1, 0, 1, 0, 1])  
pNoRefin = np.array([[[3, 2],
                      [4, 3]],
                     [[1, 6],
                      [2, 1]],
                     [[5, 4],
                      [6, 5]]])
mapNos = np.array([[4, 3], [3, 5], [5, 4], [1, 0], [0, 2], [2, 1]])
refNosEle = np.array([[[0, 0], [3, 3], [4, 4]],
                      [[3, 4], [5, 3], [4, 5]],
                      [[3, 3], [1, 1], [5, 5]],
                      [[4, 4], [5, 5], [2, 2]]])
tipoNo = np.ones((2, numNos))
tipoNo[1][noRef] = 2
funWNo = np.zeros((2, numNos))
for a in range(len(NFBS)):
    funWNo[0][NFBS[a]] = 1 
funWNo[1][noRef] = len(NHCW)
quantBaseWav = np.zeros((2,numNos))
NFG = np.zeros(2)
for i in range(2):
    CON1 = 0
    for j in range(numNos):
        quantBaseWav[i][j] = CON1
        CON1 += funWNo[i][j]
    NFG[i] = CON1
waveletsOrtog = np.zeros((2,len(NHCW),7))
waveletsOrtog[0][0][0] = 1
for i in range(len(NHCW)):
    waveletsOrtog[1][i][NHCW[i]] = 1

posNos_ = posNos
nosEle_ = np.zeros((numNos,7))
for i in range(numNos):
    nosEle_[i][0] = i
coordX = np.zeros((6,numEle))
coordY = np.zeros((6,numEle))
tipoMat = np.zeros(numEle)
nosEleF = [[0],
           [0],
           [0]]
CON2 = numNos
CON3 = CON2 + 1
CON4 = 0
for j in range(numEle):
    for k in range(3):
        coordX[k][j] = posNos[nosEle[j][k]][0]
        coordY[k][j] = posNos[nosEle[j][k]][1]
    pMedX = (np.amax(coordX[0:3, j])+np.amin(coordX[0:3, j]))/2
    pMedY = (np.amax(coordY[0:3, j])+np.amin(coordY[0:3, j]))/2
    INDA = numMat + 1
    for m in range(numMat):
        if pMedX >= matX[m][0] and pMedX <= matX[m][1] and pMedY >= matY[m][0] and pMedY <= matY[m][1]:
            INDA = m
            break
    tipoMat[j] = mat[INDA]
    nosEleAux = np.array([nosEle[j,0:3]])
    for k in range(3,6):
        coordX[k][j] = (coordX[mapNos[k][0]][j]+coordX[mapNos[k][1]][j])/2
        coordY[k][j] = (coordY[mapNos[k][0]][j]+coordY[mapNos[k][1]][j])/2
        IND1 = 0
        for l in range(CON3-1, CON2):
            if coordX[k][j] == posNos_[l][0] and coordY[k][j] == posNos_[l][1]:
                while len(nosEleAux) <= k:
                    nosEleAux = np.append(nosEleAux, [[0]])
                IND1 = 1
                nosEleAux[k] = l
        if IND1 == 0:
            while len(nosEleAux) <= k:
                nosEleAux = np.append(nosEleAux, [[0]])
            while len(posNos_) <= CON2:
                posNos_ = np.append(posNos_, [[0,0]], axis=0 )
            posNos_[CON2,0:2] = np.array([coordX[k][j],coordY[k][j]])
            nosEleAux[k] = CON2
            nosEle_[nosEle[j][mapNos[k][0]]][pNoRefin[mapNos[k][0]][0][posTri[j]]] = CON2
            nosEle_[nosEle[j][mapNos[k][1]]][pNoRefin[mapNos[k][1]][1][posTri[j]]] = CON2
            CON2 += 1
    for k in range(4):
        while len(nosEleF[0]) <= CON4:
            nosEleF = np.append(nosEleF, [[0],[0],[0]], axis=1)
        for l in range(3):
            nosEleF[l][CON4] = nosEleAux[refNosEle[k,l,posTri[j]]]
        CON4 += 1
IGFBG = np.zeros((int(NFG[0]),int(NFG[1])))
for j in range(numEle):
    for k in range(2):
        NFE[k] = funWNo[k][nosEle[j][0]]+funWNo[k][nosEle[j][1]]+funWNo[k][nosEle[j][2]]
    NFET = NFE[0]+NFE[1]
    gradFunBaseEle= np.zeros((2,int(NFET),4))
    pWvNoEl = np.zeros((int(NFET),6))
    pSoluEle = np.zeros((int(NFET),1))
    CONA=0
    for i in range(2):
        for k in range(3):
            for l in range(int(funWNo[i][nosEle[j][k]])):
                pWvNoEl[CONA][k] = waveletsOrtog[int(tipoNo[i][nosEle[j][k]])-1][l][0]
                pWvNoEl[CONA][mapNos[k][0]] = waveletsOrtog[int(tipoNo[i][nosEle[j][k]])-1][l][pNoRefin[k][0][posTri[j]]]
                pWvNoEl[CONA][mapNos[k][1]] = waveletsOrtog[int(tipoNo[i][nosEle[j][k]])-1][l][pNoRefin[k][1][posTri[j]]]
                pSoluEle[CONA] = quantBaseWav[i][nosEle[j][k]] + l
                CONA += 1
    for m in range(int(NFET)):
        gradFunBaseEle[0:2, m, 0] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][1]/2+pWvNoEl[m][3],
                                              -pWvNoEl[m][0]/2+pWvNoEl[m][2]/2+pWvNoEl[m][4]])
        gradFunBaseEle[0:2, m, 1] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][2]/2-pWvNoEl[m][3]+pWvNoEl[m][5],
                                              -pWvNoEl[m][1]/2+pWvNoEl[m][2]/2-pWvNoEl[m][3]+pWvNoEl[m][4]])
        gradFunBaseEle[0:2, m, 2] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][1]/2-pWvNoEl[m][3],
                                              -pWvNoEl[m][0]/2+pWvNoEl[m][2]/2-pWvNoEl[m][3]+pWvNoEl[m][5]])
        gradFunBaseEle[0:2, m, 3] = np.array([-pWvNoEl[m][0]/2+pWvNoEl[m][1]/2-pWvNoEl[m][4]+pWvNoEl[m][5],
                                              -pWvNoEl[m][0]/2+pWvNoEl[m][2]/2-pWvNoEl[m][4]])
    IGFBE = np.zeros((int(NFE[0]),int(NFE[1])))
    for m in range(4):
        J = np.array([[coordX[int(refNosEle[m][1][0])][j]-coordX[int(refNosEle[m][0][0])][j],
                       coordY[int(refNosEle[m][1][0])][j]-coordY[int(refNosEle[m][0][0])][j]],
                      [coordX[int(refNosEle[m][2][0])][j]-coordX[int(refNosEle[m][0][0])][j],
                       coordY[int(refNosEle[m][2][0])][j]-coordY[int(refNosEle[m][0][0])][j]]])
        gradFunBaseEleTransf1 = np.linalg.solve(J, gradFunBaseEle[:, 0:int(NFE[0]), m])
        gradFunBaseEleTransf2 = np.linalg.solve(J, gradFunBaseEle[:, int(NFE[0]):int(NFET), m])
        igfbeaux = np.transpose(gradFunBaseEleTransf1).dot(gradFunBaseEleTransf2)
        IGFBE += igfbeaux*(1/(tipoMat[j]*4*np.pi*1e-7))*np.abs(np.linalg.det(J))*0.5
    for k in range(int(NFE[0])):
        for l in range(int(NFE[1])):
            IGFBG[int(pSoluEle[k])][int(pSoluEle[int(NFE[0])+l])] += IGFBE[k][l]
#CWN=null(IGFBG);
#NFTN=size(CWN,2);
#TT=zeros(NFTN,7);
#for i=1:NFTN
#    for j=1:len(NHCW)
#        TT(i,NHCW(j))=CWN(j,i);
#for i=1:NFTN
#    TTG=zeros(CON2,1);
#    TTG(NAW)=TT(i,1);
#    for k=2:7
#        if nosEle_(NAW,k)~=0
#            TTG(nosEle_(NAW,k))=TT(i,k)+0.5*TTG(NAW);
#    figure(i)
#    trisurf(NGEF',posNos_(:,1),posNos_(:,2),TTG)