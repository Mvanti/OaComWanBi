def FGMP():
    import numpy as np
    import L

    LXDOM,LYDOM,LSI,LXCC,LYCC,SCC,LXMAT,LYMAT,VMAT,LXEXC,LYEXC,VEXC,NDOM,NCC,NMAT,NEXC = L.L()
    """CON1 irá servir para contar o número de nós
    CON2 irá servir para contar o número de elementos
    CXNI = novos pontos de X
    CYNI = novos pontos de Y
    INDNGE e INDCNI são somente auxiliares
    CNI = Coordenadas [x,y] de cada nó
    NGE = Matriz que indica [nó1, nó2, nó3] que forma cada triangulo"""
    CON1=0
    CON2=0
    CXNI=np.zeros([3])
    CYNI=np.zeros([3])
    INDNGE=1
    INDCNI=1
    CNI=np.array([[]])
    NGE=np.array([[]])
    """for j em NDOM para fazer esse processo para cada um de nossos N domínios"""
    for j in range(NDOM):
        """IXPN e IYPN são matrizes que posteriormente irão servir como auxiliares para a formação dos triangulos,
        NNYDOM e NNXDOM fazem um calculo para sabera proporção entre o tamanho da malha e dos elementos
        desejados , por exemplo se nosso x vai de [0.5,1] e nosso LSI = 0.125, NNXDOM = (1-0.5)/0.125 = 4,
        logo serão necessário dividir essa distancia em 4 elementos"""
        IXPN=[[0,LSI[j][0],LSI[j][0]],[0,LSI[j][0],0]]
        IYPN=[[0,0,LSI[j][1]],[0,LSI[j][1],LSI[j][1]]]
        NNYDOM=int(((LYDOM[j][1]-LYDOM[j][0])/LSI[j][1])+1)
        NNXDOM=int(((LXDOM[j][1]-LXDOM[j][0])/LSI[j][0])+1)
        """nosso for k e for l indicam quantos QUADRADOS (que são formados por 2 triangulos) serão formados, assim
        nosso for m indica a criação de cada triangulo do nosso domínio, nossa variavel AEEW servirá para armazenar
        a informação de quais nós formam cada um de nossos triangulos, assim, nosso for n indica que cada triangulo é
        formado por 3 nós, logo a opreação é encima de cada nó."""
        for k in range(NNYDOM-1):
            for l in range(NNXDOM-1):
                for m in range(2):
                    AEEW=np.zeros([3])
                    for n in range(3):
                        """CXNI e CYNI criam cada triangulo necessário para formar diretamente nossa malha inicial 
                        independente de o quão refinada ela é, nós começamos criando no começo de nosso domíno 
                        "LXDOM[j][0]" e vamos correndo pelo pelo nosso domínio até o final "(l)*LSI[j][0]", perceba
                        que o l veio de nossa conta de quantos triangulos eram necessários para preencher o dominio.
                        IXPN e IYPN agora servem para a formação de cada triangulo, perceba a caracteriação de suas 
                        matrizes
                        IXPN = [[0,X,X], e IYPN = [[0,0,Y], 
                                [0,X,0]]           [0,Y,Y]],
                        perceba que são 6 posições em cada matriz, isso quer dizer que são 6 pontos, logo 2 triangulos
                        distintos, aonde a primeira linha de cada representa um triangulo e a segunda linha representa
                        o segundo triangulo
                        O primeiro triangulo é formado por [0,0] [X,0] e [X,Y] enquanto o segundo triangulo é formado 
                        por [0,0] [X,Y] e [0,Y]. Fazendo as retas que ligam esses pontos percebe-se que se formou um
                        quadrado com um risco diagonal"""
                        CXNI[n]=LXDOM[j][0]+(l)*LSI[j][0]+IXPN[m][n]
                        CYNI[n]=LYDOM[j][0]+(k)*LSI[j][1]+IYPN[m][n]
                        IND1=0
                        for o in range(CON1):
                            """IND1 é um indicador que avisa o programa se o ponto já existe ou não, caso ele já exista
                            ele é gravado no vetor AEEW que servirá posteriormente para nomeação do triangulo"""
                            if CXNI[n]==CNI[o][0] and CYNI[n]==CNI[o][1]:
                                IND1=1
                                AEEW[n]=o
                                break
                        """Caso o ponto não exista ainda ele então é nomeado como novo ponto, gravado no nosso vetor CNI
                        como um novo nó da minha malha e é gravado no vetor AEEW para nomeação do triangulo"""
                        if IND1==0:
                            AEEW[n]=CON1
                            CNIL=np.array([CXNI[n],CYNI[n]])
                            INDCNI-=CON1
                            CNI=np.append(CNI,[CNIL],axis=INDCNI)
                            INDCNI=CON1+1
                            CON1+=1
                    """Depois dos 3 nós (novos ou não) serem gravados no vetor AEEW, o vetor AEEW é gravado no nosso vetor
                    NGE como um novo elemento da malha"""
                    INDNGE-=CON2
                    NGE=np.append(NGE,[AEEW],axis=INDNGE)
                    INDNGE=CON2+1
                    CON2+=1
    """Calculado todos meus pontos e elementos que formam minha malha, chega a hora de marcar nosso CON1 na variavel NNG (numero de nós)
    marcar CON2 na variavel NEL (numero de elementos) e criar o vetor TE que será a excitação de cada elemento, sendo que 
    o nosso quadrado será sempre formado por um triagulo de TE = 1 e um TE = 2 (não sei por que)
    TE = 1 são os triangulos de baixo enquanto os triangulos TE = 2 são os triangulos de cima"""
    NNG=CON1
    NEL=CON2
    TE=np.zeros([NEL])
    for j in range(0,NEL,2):
        TE[j]=1
        TE[j+1]=2
    """NMN = numero de elementos nos quais tal nó participa
    TME = tipo de material de cada elemento
    TEE = tipo de excitação de cada elemento"""
    NMN=np.zeros([NNG])
    TME=np.zeros([NEL])
    TEE=np.zeros([NEL])
    """for j testa cada um dos meus elementos, for k testa cada um dos nós de cada elemento sendo que nosso "IND" indicará de 
    qual nós estamos falando, adicionando para cada vez que um nó aparece um novo elemento que ele compoe "NMN +1" e gravando 
    sua coordenada X e Y"""
    for j in range(NEL):
        for k in range(3):
            IND= NGE[j][k]
            NMN[int(IND)]+= 1
            CXNI[k]=CNI[int(IND)][0]
            CYNI[k]=CNI[int(IND)][1]
        """Em posse das suas coordenadas o programa calcula o ponto central x e y e INDA indica novamente
        o numero de materiais (desnecessario, inclusive será sobreescrito no proximo passo)"""
        PEEX=(np.amax(CXNI)+np.amin(CXNI))/2
        PEEY=(np.amax(CYNI)+np.amin(CYNI))/2
        INDA=NMAT
        """O programa testa para ver em qual dominío ou região de material meu elemento esta anexo, perceba que ele testa
        se o x e o y estão dentro dos limites do material, quando entrar o material equivalente INDA recebe o indicador do
        material"""
        for m in range(NMAT):
            if PEEX>=LXMAT[m][0] and PEEX<=LXMAT[m][1] and PEEY>=LYMAT[m][0] and PEEY<=LYMAT[m][1]:
                INDA=m
                break
        """Depois de encontrar e anotar de qual material pertence o elemento é gravado no nosso vetor TME
        Logo após é feito um processo equivalente para excitação"""
        TME[j]=VMAT[INDA]
        INDB=NEXC
        for m in range(NEXC):
            if PEEX>=LXEXC[m][0] and PEEX<=LXEXC[m][1] and PEEY>=LYEXC[m][0] and PEEY<=LYEXC[m][1]:
                INDB=m
                break
        TEE[j]=VEXC[INDB]      
    return CXNI,CYNI,INDNGE,INDCNI,IXPN,IYPN,NNYDOM,NNXDOM,AEEW,CNI,NEL,CNIL,NGE,NMN,NNG,TE,TEE,TME,IND1,IND,PEEX,PEEY,INDA
