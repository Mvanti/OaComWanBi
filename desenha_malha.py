
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:49:36 2014

@author: Leo kewitz
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

il = lambda l, offset=0: zip(np.arange(len(l))+offset,l)

class Node:
    def __init__(self,postring,i):
        x, y = postring.strip().split()
        self.x, self.y = [float(x), float(y)]
        self.i = i

class Element:
    def __init__(self,nos,i):
        args = nos.strip().split()
        self.nodes = [int(a) for a in args[0:3]]
        self.material = int(args[3])
        self.source = float(args[4])

class Mesh:
    def __init__(self, filepath):
        with open(filepath) as f:
            ls = f.readlines()
            l = ls[0].strip().split()
            ns = int(l[0])
            es = int(l[1])
            self.nodes = [Node(a[1],a[0]) for a in il(ls[1:ns+1],1)]
            self.elements = [Element(a[1],a[0]) for a in il(ls[ns+1:ns+es+1],1)]
    
    def getNode(self, index):
    # Procura todos os nós com o índice == 'index' retornando um list
        nodes = [n for n in self.nodes if n.i == index]
        #Verifica se tem mais de um nó (deveria existir só um)
        if len(nodes) == 1:
            return nodes[0]
            # Se ele não existir ou houver mais de um com o mesmo índice gera um erro
        else:
             return nodes[1]
    
    def plotMesh(self):
        cbase = 'green white  blue coral violet navy brown orange beige'.split()
        fig, ax = plt.subplots()
        for e in self.elements:
            x = []
            y = []
            for ni in e.nodes:
                n = self.getNode(ni)
                x.append(n.x)
                y.append(n.y)
                c = cbase[e.material] if e.source == 0.0 else 'grey'  # Cinza se for fonte.
                p = Polygon(list(zip(x,y)), closed=True, fc=c, ec='k', alpha=.17, linewidth=.3)
                ax.add_patch(p)
        ax.axis('image')

