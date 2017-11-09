def L():
    import numpy as np
    
    """ Primeiramente devemos atribuir o domínio, LXDOM determina os limites x de cada um dos meus domínios,
    LXDOM = [xinicial(dominio 1),xfinal],[xinicial(dominio 2),xf..., LYDOM faz o processo equivalente com o
    Y e o LSI determina qual irá ser o nivel de refinamento inicial da minha malha 
    [x dominio 1, y dominio 1],[x domonio 2,..."""
    
    LXDOM=np.array([[0,1],[0,0.5]])
    LYDOM=np.array([[0,0.5],[0.5,1]])
    LSI=np.array([[0.125,0.125],[0.125,0.125]])
    
    """ LXCC e LYCC fazem o mesmo processo que LXDOM e LYDOM, só que ao invés de determinar os limites dos
    domínios, determinam os limites das retas com condições de contorno, sendo que 
    LXCC = [xinical(reta 1),xfinal],[xinicial(reta 2),x..., Para cada reta de condição de contorno nomeada
    SCC irá indicar o valor atribuido"""
    
    LXCC=np.array([[0.5,0.5],[0.5,1],[0,1],[0,0],[1,1],[0,0.5]])
    LYCC=np.array([[0.5,1],[0.5,0.5],[0,0],[0,1],[0,0.5],[1,1]])
    SCC=np.array([0,0,0,0,0,0])
    
    
    """LXMAT e LYMAT indicam o domínio ou região que cada material ocupa, sendo 
    LXMAT = [xinicial(material 1), xfinal], [xinicial(material 2), x... sendo que VMAT irá indicar a 
    caracterização de cada matérial"""
    
    LXMAT=np.array([[0,1]])
    LYMAT=np.array([[0,1]])
    VMAT=np.array([1])
    
    """LXEXC e LYEXC inciam o domínio ou região que cada excitação ocoore, sendo
    LXEXC = [xinicial(excitação 1), xfinal], [xinicial(excitação 2), x... sendo que VEXC irá indicar a
    caracterização de cada excitação"""
    
    LXEXC=np.array([[0,1]])
    LYEXC=np.array([[0,1]])
    VEXC=np.array([1])
    
    """NDOM = número de domínios (desnecessário);
    NCC = número de condições de contorno (desnecessário);
    NMAT = número de materiais (desnecessário);
    NEXC = número de excitações (desnecessário);"""
    NDOM=len(LSI);
    NCC=len(SCC)
    NMAT=len(VMAT)
    VMAT=np.append(VMAT,1)
    NEXC=len(VEXC)
    VEXC=np.append(VEXC,0)
    
    return LXDOM,LYDOM,LSI,LXCC,LYCC,SCC,LXMAT,LYMAT,VMAT,LXEXC,LYEXC,VEXC,NDOM,NCC,NMAT,NEXC