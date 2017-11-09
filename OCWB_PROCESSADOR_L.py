# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:00:19 2016

@author: Nemesio Sopelsa
"""
def OPL(NNG,NMN,CNI):
    import numpy as np
    
    """RNE servirá para fazer o mapeamento do triangulo apos ser refinado
    3
    |\  
    | \ 
   5|__\6
    |\  |\
    | \ | \
    |__\|__\
    1   4   2
    
    RNE[0]= linha entre 5 e 4  {
    RNE[1] = linha entre 4 e 6 { Triangulo Refinado
    RNE[2] = linha entre 6 e 5 {
    RNE[3] = linha entre 2 e 1 {
    RNE[4] = linha entre 1 e 3 { Triangulo Original
    RNE[5] = linha entre 3 e 2 {"""
    EFN2NC = np.array([[1,4,5,5,4,2,5,2,3],[1,6,3,1,4,6,4,2,6],[1,6,5,1,2,6,5,6,3]])
    EFN1NC = np.array([[1,4,3,4,2,3],[1,2,5,5,2,3],[1,2,6,1,6,3]])
    REES = np.array([[[1,1],[4,4],[5,5]],[[4,5],[6,4],[5,6]],[[4,4],[2,2],[6,6]],[[5,5],[6,6],[3,3]]])
    RNE = np.array([[5,4],[4,6],[6,5],[2,1],[1,3],[3,2]])
    PPNTE = np.array([[[4,3],[5,4]],[[2,7],[3,2]],[[6,5],[7,6]]])
    FBEL = np.array([1-1/3-1/3,1/3,1/3])
    GFBEL = np.array([[-1,1,0],[-1,0,1]])
    TEES = np.array([[1,2],[2,1],[1,2],[1,2]])
    ZZZ = np.array([2,1])
    
    
    """Tudo que é nomeado antes desse comentário na verdade está nomeado desnecessáriamente e não é utilizado nessa parte
    do programa, mas que provavelmente são propriedades das wavelets que serão usadas no programa principal.
    
    TN é um vetor que servirá para indicar o tipo de cada nó da minha malha, levando em consideração a localização dele
    na malha e o número de elementos em que ele participa (NMN), perceba que o valor padrão de TN é 1, que reprenseta os
    nós contidos no meio da malha que fazem parte de 6 elementos simultaneamente, para cada tipo de nó diferente é 
    realizado uma série de "ifs" para determinar a que grupo "TN" ele pertence"""
    TN = np.ones((NNG,1))
    for j in range(NNG):
        if NMN[j] != 6:
            if NMN[j] == 4:
                TN[j] = 10
            elif NMN[j] == 3:
                if CNI[j][0] == 0:
                    TN[j] = 4
                elif CNI[j][1] == 0:
                    TN[j] = 2
                elif CNI[j][1] == 0.5 or CNI[j][1] == 1:
                    TN[j] = 3
                else:
                    TN[j] = 5
            elif NMN[j] == 2:
                if CNI[j][0] == 0:
                    TN[j] = 6
                else:
                    TN[j] = 9
            elif CNI[j][0] == 0:
                TN[j] = 8
            else:
                TN[j] = 7
    """T1, T2, T3, T4 e T5 na verdade são wavelets já pre-calculadas para tipos especificos de TN, por exemplo nós
    do tipo TN = 1 geram 3 walvelets linearmente indepentes, e como essas wavelets só dependem do espaço, tais wavelets
    já podem ser pré calculadas e guardadas num banco de dados, assim T1 será composto por um vetor que carrega três 
    conjuntos de indices para minhas wavelets
    NFTN indica o numero que funções presentes em cada Tipo de Nó
    PWTN é uma matriz que carrega todos nossos vetore TN usados do nosso banco de dados
    NGTN indica como vai ser o comportar (qual TN vai pertencer) os novos nós vindo do refinamento da malha,
    
    Considere no desenho a seguir o nó 1 sendo o elemento original com TN já defindo, digamos que seu TN = 2
    NGTN[1] = [2,2,1,1,2,0,0], isso quer dizer que os pontos 1, 2 e 5 do meu desenho terão TN = 2, 
    meus pontos 3 e 4 terão TN = 1 
    e meus pontos 6 e 7 terão TN = 0 (não existem)

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
    T1 = np.array([[-0.3675,0.2208,0.3722,0.4432,0.5142,0.3628,0.2918],[-0.0654,0.6177,0.4844,-0.0012,-0.4869,-0.3537,0.1320],[0.0593,-0.3595,0.4317,0.3363,0.2409,-0.5503,-0.4549]])
    NFTN = np.array([[len(T1)]])
    PWTN = np.zeros([5,len(T1),len(T1[0])])
    PWTN[0,0:3,0:7]=T1
    NGTN = np.array([[1,1,1,1,1,1,1]])
    T2 = np.array([[0,0,0.8944,0.4472,0,0,0]])
    NFTN = np.append(NFTN,[[len(T2)]],axis = 0)
    PWTN[1,0:1,0:7]=T2
    NGTN = np.append(NGTN,[[2,2,1,1,2,0,0]],axis = 0)
    T3 = [[0,0,0,0,0,-0.8944,-0.4472]] 
    NFTN = np.append(NFTN,[[len(T3)]],axis = 0)
    PWTN[2,0:1,0:7]=T3
    NGTN = np.append(NGTN,[[3,3,0,0,3,1,1]],axis = 0)
    T4 = [[0,0,0,-0.4472,-0.8944,0,0]]
    NFTN = np.append(NFTN,[[len(T4)]],axis = 0)
    PWTN[3,0:1,0:7]=T4
    NGTN = np.append(NGTN,[[4,0,4,1,1,4,0]],axis = 0)
    T5 = [[0,-0.8944,0,0,0,0,-0.4472]]
    NFTN = np.append(NFTN,[[len(T5)]],axis = 0)
    PWTN[4,0:1,0:7]=T5
    NGTN = np.append(NGTN,[[5,1,5,0,0,5,1]],axis = 0)
    NFTN = np.append(NFTN,[[0]],axis = 0)
    NGTN = np.append(NGTN,[[6,0,4,1,2,0,0]],axis = 0)
    NFTN = np.append(NFTN,[[0]],axis = 0)
    NGTN = np.append(NGTN,[[7,2,5,0,0,0,0]],axis = 0)
    NFTN = np.append(NFTN,[[0]],axis = 0)
    NGTN = np.append(NGTN,[[8,0,0,0,3,4,0]],axis = 0)
    NFTN = np.append(NFTN,[[0]],axis = 0)
    NGTN = np.append(NGTN,[[9,3,0,0,0,5,1]],axis = 0)
    NFTN = np.append(NFTN,[[0]],axis = 0)
    NGTN = np.append(NGTN,[[10,1,5,0,3,1,1]],axis = 0)
    
    
    """Depois de organizar minhas matrizes e vetores, crio um vetor NFN que irá carregar quantas wavelts cada nó criou"""
    NFN = np.array(NFTN[int(TN[0])])
    for i in range(1,NNG):
        NFN = np.append(NFN,NFTN[int(TN[i]-1)],axis = 0)
        
        
    return EFN2NC,EFN1NC,REES,RNE,PPNTE,FBEL,GFBEL,TEES,ZZZ,TN,T1,T2,T3,T4,T5,NFTN,PWTN,NGTN,NFN



