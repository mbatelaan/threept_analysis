#!/usr/bin/env python
# from sympy.matrices import Matrix,eye
import numpy as np

gamma1 = np.array(([0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]))
gamma2 = np.array(([0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]))
gamma3 = np.array(([0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]))
gamma4 = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]))
# gamma2 = np.array(([0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]))
# gamma3 = np.array(([0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]))
# gamma4 = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]))
# gamma5 = np.array(([0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]))
gamma5 = np.array(([0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0]))
gammaI = np.identity(4)
sigma41 = 1.0j * np.matmul(gamma4, gamma1)
sigma42 = 1.0j * np.matmul(gamma4, gamma2)
sigma43 = 1.0j * np.matmul(gamma4, gamma3)
sigma12 = 1.0j * np.matmul(gamma1, gamma2)
sigma13 = 1.0j * np.matmul(gamma1, gamma3)
sigma14 = -1.0 * sigma41
sigma21 = -1.0 * sigma12
sigma23 = 1.0j * np.matmul(gamma2, gamma3)
sigma24 = -1.0 * sigma42
sigma31 = -1.0 * sigma13
sigma32 = -1.0 * sigma23
sigma34 = -1.0 * sigma43
# GammaUP = 0.5 * np.matmul(gammaI, gamma4)
GammaUP = 0.5 * (gammaI + gamma4)
# GammaX = GammaUP * gamma1 * gamma5 * 1.0j
# GammaY = GammaUP * gamma2 * gamma5 * 1.0j
GammaX = -1.0j * np.matmul(GammaUP, np.matmul(gamma1, gamma5))
GammaY = -1.0j * np.matmul(GammaUP, np.matmul(gamma2, gamma5))
GammaZ = -1.0j * np.matmul(GammaUP, np.matmul(gamma3, gamma5))
# GammaZ = -1.0j * np.matmul(np.matmul(GammaUP, gamma3), gamma5)
# GammaX = GammaUP * gamma5 * gamma1 * 1.0j
# GammaY = GammaUP * gamma5 * gamma2 * 1.0j
# GammaZ2 = GammaUP * gamma5 * gamma3 * 1.0j
# GammaZ = -0.5j * (gamma1 * gamma2 + gamma3 * gamma5)
GammaP5 = GammaUP * gamma5
GammaXY = GammaUP * gamma2 * gamma1 * 1.0j

QDPGammaxx = {
    0: gammaI,
    1: gamma1,
    2: gamma2,
    3: gamma1 * gamma2,
    4: gamma3,
    5: gamma1 * gamma3,
    6: gamma2 * gamma3,
    7: gamma4 * gamma5,
    8: gamma4,
    9: gamma1 * gamma4,
    10: gamma2 * gamma4,
    11: gamma3 * gamma5,
    12: gamma3 * gamma4,
    13: gamma2 * gamma5,
    14: gamma1 * gamma5,
    15: gamma5,
}
QDPGammaInd = {1: 0, 2: 1, 4: 2, 8: 3}
QDPGammaInd5 = {14: 1, 13: 2, 11: 3, 7: 0}
