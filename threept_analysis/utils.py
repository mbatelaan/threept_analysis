from numpy import (
    zeros,
    size,
    sort,
    exp,
    average,
    split,
    append,
    multiply,
    sum,
    sqrt,
    linspace,
)
from numpy import array as array

# from BootStrap import BootStrap1


def ReadEvxpt(file, par):
    FF = []
    f = open(file)
    for line in f:
        strpln = line.rstrip()
        if len(strpln) > 0:
            if (
                strpln[0] == "+"
                and strpln[1] == "F"
                and strpln[2] == "I"
                and strpln[5] == "0"
            ):
                tmp = f.readline().split()
                nff = int(tmp[2])
                tmp = f.readline()
                tmp = f.readline()
                tmp = f.readline()
                tmp = f.readline()
                tmp = f.readline()
                tmp = f.readline()
                tmp = f.readline().split()
                boots = int(tmp[3])
                tmp = f.readline()
                FF.append(BootStrap1(boots, 68))
                for iff in range(nff):
                    tmp = f.readline().split()
                    if par == iff:
                        FF[-1].Avg = float(tmp[2])
                        FF[-1].Std = float(tmp[3])
                    # if iff==(nff-1):
                    #     baddata=append(baddata,float(tmp[2]))
                tmp = f.readline().split()
                if tmp[0] == "+NUmbers=" + str(boots):
                    for iboot in range(boots):
                        tmp = f.readline().split()
                        FF[-1].values[iboot] = float(tmp[2 * par + 4])
    f.close()
    return FF


def ReadEvxptdump(file, par, confs, times, boots):
    # par should be 0 or 1 depending on whether the real or imaginary part is chosen (0=R,1=I).
    f = open(file)
    FF = []
    G = zeros(shape=(confs, times, 2))
    for line in f:
        strpln = line.rstrip()
        if len(strpln) > 0:
            if (
                strpln[0] == "+"
                and strpln[1] == "R"
                and strpln[2] == "D"
                and int(strpln[4]) == 0
            ):
                for iff in range(confs):
                    for nt in range(times):
                        tmp = f.readline().split()
                        G[iff, nt, 0] = tmp[1]
                        G[iff, nt, 1] = tmp[2]
    f.close()
    for j in range(times):
        FF.append(BootStrap1(boots, 68))
        FF[-1].Import(G[:, j, par])
        FF[-1].Stats()
    return FF
