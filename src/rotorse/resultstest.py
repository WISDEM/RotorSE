import numpy as np

ELTx_1 = np.loadtxt('ELTx.txt')
ELTx_2 = np.zeros(18 * 7)
ELTy_1 = np.loadtxt('ELTy.txt')
ELTy_2 = np.zeros(18 * 7)
Carbony_1 = np.loadtxt('Carbony.txt')
Carbony_2 = np.zeros(18 * 7)
Triaxx_1 = np.loadtxt('Triaxx.txt')
Triaxx_2 = np.zeros(18 * 7)
life_time_damage_1 = np.loadtxt('life_time_damage.txt')
life_time_damage_2 = np.zeros(18 * 7)
critical_deflection_1 = np.loadtxt('critical_deflection.txt')
critical_deflection_2 = np.zeros(18 * 7)
for i in range(0, 18):
    for j in range(0, 7):
        ELTx_2[i * 7 + j] = ELTx_1[i, j]
        ELTy_2[i * 7 + j] = ELTy_1[i, j]
        Carbony_2[i * 7 + j] = Carbony_1[i, j]
        Triaxx_2[i * 7 + j] = Triaxx_1[i, j]

np.savetxt('ELTx_new.txt', ELTx_2)
np.savetxt('ELTy_new.txt', ELTy_2)
np.savetxt('Carbony_new.txt', Carbony_2)
np.savetxt('Triaxx_new.txt', Triaxx_2)