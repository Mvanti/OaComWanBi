def SM(posNos,ampNos,nosEle,valorContorno,noContorno):
    arq = open('Refin.malha', 'w')
    arq.write(str(len(posNos)))
    arq.write("\t")
    arq.write(str(len(nosEle)))
    arq.write("\t")
    arq.write(str(len(valorContorno)))
    arq.write("\n")
    for i in range(len(posNos)):
        arq.write(str(posNos[i][0]))
        arq.write("\t")
        arq.write(str(posNos[i][1]))
        arq.write("\n")
    for i in range(len(nosEle)):
        arq.write(str(int(nosEle[i][0] + 1)))
        arq.write("\t")
        arq.write(str(int(nosEle[i][1] + 1)))
        arq.write("\t")
        arq.write(str(int(nosEle[i][2] + 1)))
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
    for i in range(len(ampNos)):
        arq.write(str(ampNos[i][0]))
        arq.write("\t")
        arq.write("\n")
    arq.close()