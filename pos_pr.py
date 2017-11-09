# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12014

@author: MV
"""
import sys

import desenha_malha as ds
import matplotlib.pyplot as plt
import matplotlib.tri as tri

try:
    infilename=sys.argv[1]
except:
    print ("Usage:"), sys.argv[0], ("infile"); sys.exit(1)

infilename1=infilename+".malha"
infilename2=infilename+".resu"

#arquivo .malha
m = ds.Mesh(infilename1)

# arquivo de resultados
ifile=open(infilename2,'r')

v=[]
lines=ifile.readlines()

for line in lines:
    v.append(float(line))

x = []
y = []
z = []
for n in m.nodes:
    x.append(n.x)
    y.append(n.y)
    z.append(v[n.i-1])

m.plotMesh()    
tri.Triangulation(x, y)
plt.tricontour(x, y, z, 20, linewidths=0.5)
plt.tricontourf(x, y, z, 20, alpha=1,cmap=plt.cm.coolwarm)
plt.title('EquiPotential Plot')
plt.colorbar()
plt.show()
