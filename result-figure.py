import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
# x is the size of dataset: CORR (c), u, INDF (u), ANTI (a)
x = [1000, 3000, 5000, 7000, 9000]
# BSSP and FSSP
# BSSP: bc1, bu1, ba1 is the process time, while bc2, bu2, ba2 is the query time
bc1 = [7.634, 23.333, 38.434, 53.506, 67.909]
bu1 = [6.984, 21.282, 48.686, 58.165, 80.815]
ba1 = [15.413, 29.320, 44.100, 62.852, 80.703]
bc2 = [43.161, 132.170, 216.781, 303.955, 452.139]
bu2 = [56.557, 165.669, 237.253, 346.832, 438.630]
ba2 = [48.558, 170.933, 248.067, 352.676, 446.624]
# FSSP: fc1, fu1, fa1 is the process time, while fc2, fu2, fa2 is the query time
fc1 = [7.08, 25.45, 35.63, 51.91, 66.21]
fu1 = [7.009, 21.064, 37.49, 52.39, 67.55]
fa1 = [7.140, 22.32, 40.289, 60.173, 74.365]
fc2 = [1967.32, 9397.796, 12006.25, 18929.55, 21051.93]
fu2 = [2042.537, 9534.585, 14889.13, 17509.35, 22798.78]
fa2 = [2052.421, 7408.72, 10974.647, 15516.659, 20830.13]
# FSSP: fcA, fuA, faA is the communication of server_1, while fcB, fuB, faB is the communication of server_2
fcA = [236.51, 511.96, 809.93, 1218.5, 1301.51]  # MB
fcB = [123.02, 264.04, 418.09, 628.05, 673.46]
fuA = [334.74, 654.05, 870.36, 1049.31, 1264.52]
fuB = [172.62, 335.81, 448.61, 542.60, 654.78]
faA = [312.81, 476.44, 744.94, 880.12, 1375.48]
faB = [161.17, 246.09, 378.86, 457.15, 710.81]
# dataset: diabets and obesity (do)
# BSSP: bdo1 is the process time, while bdo2 is the query time
bdo1 = [7.04, 7.14]
bdo2 = [46.07, 50.70]
# FSSP: fdo1 is the process time, while fdo2 is the query time
fdo1 = [7.24, 7.20]
fdo2 = [499.35, 1663.57]
# FSSP: fdoA is the communication of server_1, while fdoB is the communication of server_2
fdoA = [34.43, 111.58]  # MB
fdoB = [19.19, 58.20]
# compute the entire time in the BSSP and FSSP
bc2 = np.array(bc2) + np.array(bc1)
fc2 = np.array(fc2) + np.array(fc1)
bu2 = np.array(bu2) + np.array(bu1)
fu2 = np.array(fu2) + np.array(fu1)
ba2 = np.array(ba2) + np.array(ba1)
fa2 = np.array(fa2) + np.array(fa1)
bdo2 = np.array(bdo2) + np.array(bdo1)
fdo2 = np.array(fdo2) + np.array(fdo1)

# Ours scheme
# share
# Ours share time in the different datasets
sc = [3.48, 27.13, 56.23, 83.37, 127.7]
su = [4.28, 34.69, 79.04, 115.39, 172.31]
sa = [3.79, 10.28, 57.47, 85.96, 131.55]
sdo = [0.008, 0.026]
# process
# Ours process time in the different datasets
pc = [2516.9, 15683.2, 33316.7, 50617.4, 71213.8]
pu = [3109.3, 19548.1, 45748.6, 76575.7, 103189.2]
pa = [2525.1, 15403.3, 33305.2, 56271.7, 66462.1]
pdo = [8.47, 27.59]
# query
# Ours quadrant query time in the different datasets
qc = [1.535, 4.636, 7.294, 9.918, 11.584]
qu = [1.638, 5.438, 9.417, 13.708, 17.330]
qa = [1.586, 4.756, 7.645, 10.242, 12.441]
qdo = [0.106, 0.090]
# Ours quadrant query time in the different datasets with multiple threads
qcT = [0.247, 0.623, 0.937, 1.331, 1.528]
quT = [0.202, 0.577, 1.004, 1.549, 1.961]
qaT = [0.222, 0.577, 0.890, 1.220, 1.491]
qdoT = [0.079, 0.077]

# dynamic
# Ours dynamic query time in the different datasets
qcD = [9.43, 16.02, 23.08, 26.15, 29.97]
quD = [10.53, 16.02, 21.91, 31.83, 36.76]
qaD = [11.72, 19.15, 22.15, 28.43, 31.86]
qdoD = [1.68, 2.06]
# Ours dynamic query time in the different datasets with multiple threads
qcDT = [3.15, 3.94, 4.44, 5.62, 6.11]
quDT = [3.24, 4.12, 5.84, 6.85, 7.17]
qaDT = [3.34, 4.29, 5.72, 6.35, 6.47]
qdoDT = [0.49, 0.71]
# Ours dynamic query communication in the different datasets with multiple threads
qcA = [10.30, 19.33, 25.94, 30.08, 32.62]  # MB
quA = [10.46, 19.93, 27.61, 33.57, 38.75]
qaA = [11.09, 20.42, 26.26, 30.35, 32.55]
qdoA = [0.83, 1.32]
qcB = [46.07, 66.48, 79.17, 86.62, 88.37]
quB = [43.62, 62.18, 74.72, 83.08, 92.23]
qaB = [50.50, 73.52, 82.13, 87.39, 88.95]
qdoB = [4.80, 9.26]


# print(fc2)
# print(fu2)
# print(fa2)
# plt.legend(loc=*)
# 0: ‘best';1: ‘upper right';2: ‘upper left';3: ‘lower left';4: ‘lower right';5: ‘right'
# 6: ‘center left';7: ‘center right';8: ‘lower center';9: ‘upper center';10: ‘center'

# Figure 1
# plt.xticks(x)
# plt.xlabel('number of tuples n')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([0.1, 1, 10, 100, 10000, 100000])
# plt.ylim(0.1, 1000001)
# plt.plot(x, pc, color='blue', linestyle='dotted', label='Ours-SDG', marker='^')
# plt.plot(x, qc, color='g', linestyle='-.', label='Ours-SPE', marker='s')
# plt.plot(x, qcT, color='orange', linestyle='-.', label='Ours-SPE (Thread)', marker='h')
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\TimeCORRQuadrant.pdf', bbox_inches='tight')
# plt.show()
# Figure 2
# plt.xticks(x)
# plt.xlabel('number of tuples n')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([0.1, 1, 10, 100, 10000, 100000])
# plt.ylim(0.1, 1000001)
# plt.plot(x, pu, color='blue', linestyle='dotted', label='Ours-SDG', marker='^')
# plt.plot(x, qu, color='g', linestyle='-.', label='Ours-SPE', marker='s')
# plt.plot(x, quT, color='orange', linestyle='-.', label='Ours-SPE (Thread)', marker='h')
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\TimeINDEQuadrant.pdf', bbox_inches='tight')
# plt.show()
# Figure 3
# plt.xticks(x)
# plt.xlabel('number of tuples n')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([0.1, 1, 10, 100, 10000, 100000])
# plt.ylim(0.1, 1000001)
# plt.plot(x, pa, color='blue', linestyle='dotted', label='Ours-SDG', marker='^')
# plt.plot(x, qa, color='g', linestyle='-.', label='Ours-SPE', marker='s')
# plt.plot(x, qaT, color='orange', linestyle='-.', label='Ours-SPE (Thread)', marker='h')
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\TimeANTIQuadrant.pdf', bbox_inches='tight')
# plt.show()
# Figure 4
# xt = np.arange(2, 3, 0.8)
# total_width, n = 0.6, 4
# width = total_width / n
# xt = xt - (total_width - width) / 2
# xdo = ['diabetes', 'obesity']
# plt.xticks(xt, xdo)
# plt.xlabel('n = 1000')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([0.01, 0.1, 1, 10, 100])
# plt.ylim(0.01, 100)
# plt.bar(xt - width, pdo, width=width, label="Ours-SDG", color="b")
# plt.bar(xt, qdo, width=width, label="Ours-SPE", color="g")
# plt.bar(xt + width, qdoT, width=width, label="Ours-SPE (Thread)", color="orange")
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\TimeREALQuadrant.pdf', bbox_inches='tight')
# plt.show()
# Figure 5
# plt.xticks(x)
# plt.xlabel('number of tuples n')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([1, 10, 100, 1000, 10000, 100000])
# plt.ylim(1, 300000)
# plt.plot(x, fc2, color='blue', linestyle='dotted', label='Liu-FSSP', marker='^')
# plt.plot(x, bc2, color='red', linestyle='-', label='Liu-BSSP', marker='o')
# plt.plot(x, qcD, color='g', linestyle='-.', label='Ours-SPE', marker='s')
# plt.plot(x, qcDT, color='orange', linestyle='--', label='Ours-SPE (Thread)', marker='d')
# plt.legend(loc=2)
# plt.savefig('TimeCORR.pdf', bbox_inches='tight')
# plt.show()
# Figure 6
# plt.xticks(x)
# plt.xlabel('number of tuples n')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([1, 10, 100, 1000, 10000, 100000])
# plt.ylim(1, 300000)
# plt.plot(x, fu2, color='blue', linestyle='dotted', label='Liu-FSSP', marker='^')
# plt.plot(x, bu2, color='red', linestyle='-', label='Liu-BSSP', marker='o')
# plt.plot(x, quD, color='g', linestyle='-.', label='Ours-SPE', marker='s')
# plt.plot(x, quDT, color='orange', linestyle='--', label='Ours-SPE (Thread)', marker='d')
# plt.legend(loc=2)
# plt.savefig('\TimeINDE.pdf', bbox_inches='tight')
# plt.show()
# Figure 7
# plt.xticks(x)
# plt.xlabel('number of tuples n')
# plt.ylabel('time (s)')
# plt.yscale('log')
# plt.yticks([1, 10, 100, 1000, 10000, 100000])
# plt.ylim(1, 300000)
# plt.plot(x, fa2, color='blue', linestyle='dotted', label='Liu-FSSP', marker='^')
# plt.plot(x, ba2, color='red', linestyle='-', label='Liu-BSSP', marker='o')
# plt.plot(x, qaD, color='g', linestyle='-.', label='Ours-SPE', marker='s')
# plt.plot(x, qaDT, color='orange', linestyle='--', label='Ours-SPE (Thread)', marker='d')
# plt.legend(loc=2)
# plt.savefig('TimeANTI.pdf', bbox_inches='tight')
# plt.show()
# Figure 8
# xt = np.arange(2, 3, 0.8)
# total_width, n = 0.6, 4
# width = total_width / n
# xt = xt - (total_width - width) / 2
# xdo = ['diabetes', 'obesity']
# plt.xticks(xt, xdo)
# plt.xlabel('n = 1000')
# plt.yscale('log')
# plt.ylim(0.1, 20000)
# plt.ylabel('time (s)')
# plt.bar(xt - width, fdo2, width=width, label="Liu-FSSP", color="b")
# plt.bar(xt, bdo2, width=width, label="Liu-BSSP", color="r")
# plt.bar(xt + width, qdoD, width=width, label="Ours-SPE", color="g")
# plt.bar(xt + 2*width, qdoDT, width=width, label="Ours-SPE (Thread)", color="orange")
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\TimeREAL.pdf', bbox_inches='tight')
# plt.show()
# Figure 9
# xt = np.arange(2, 7, 1)
# total_width, n = 0.8, 4
# width = total_width / n
# xt = xt - (total_width - width) / 2
# plt.xlabel('number of tuples n')
# plt.xticks(xt, x)
# plt.yticks(np.arange(0, 1501, 500))
# plt.ylim(0, 1501)
# plt.ylabel('cost (MB)')
# plt.bar(xt - width, fcA, width=width, label="Liu-FSSP (CS)", color="b")
# plt.bar(xt, fcB, width=width, label="Liu-FSSP (ES)", color="darkblue")
# plt.bar(xt + width, qcA, width=width, label="Ours-SPE (CS)", color="g")
# plt.bar(xt + 2*width, qcB, width=width, label="Ours-SPE (ES)", color="darkgreen")
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\CostCORR.pdf', bbox_inches='tight')
# plt.show()
# Figure 10
# xt = np.arange(2, 7, 1)
# total_width, n = 0.8, 4
# width = total_width / n
# xt = xt - (total_width - width) / 2
# plt.xlabel('number of tuples n')
# plt.xticks(xt, x)
# plt.yticks(np.arange(0, 1501, 500))
# plt.ylim(0, 1501)
# plt.ylabel('cost (MB)')
# plt.bar(xt - width, fuA, width=width, label="Liu-FSSP (CS)", color="b")
# plt.bar(xt, fuB, width=width, label="Liu-FSSP (ES)", color="darkblue")
# plt.bar(xt + width, quA, width=width, label="Ours-SPE (CS)", color="g")
# plt.bar(xt + 2*width, quB, width=width, label="Ours-SPE (ES)", color="darkgreen")
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\CostINDE.pdf', bbox_inches='tight')
# plt.show()
# Figure 11
# xt = np.arange(2, 7, 1)
# total_width, n = 0.8, 4
# width = total_width / n
# xt = xt - (total_width - width) / 2
# plt.xlabel('number of tuples n')
# plt.xticks(xt, x)
# plt.yticks(np.arange(0, 1501, 500))
# plt.ylim(0, 1501)
# plt.ylabel('cost (MB)')
# plt.bar(xt - width, faA, width=width, label="Liu-FSSP (CS)", color="b")
# plt.bar(xt, faB, width=width, label="Liu-FSSP (ES)", color="darkblue")
# plt.bar(xt + width, qaA, width=width, label="Ours-SPE (CS)", color="g")
# plt.bar(xt + 2*width, qaB, width=width, label="Ours-SPE (ES)", color="darkgreen")
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\CostANTI.pdf', bbox_inches='tight')
# plt.show()
# Figure 12
# xt = np.arange(2, 3, 0.8)
# total_width, n = 0.6, 4
# width = total_width / n
# xt = xt - (total_width - width) / 2
# xdo = ['diabetes', 'obesity']
# plt.xticks(xt, xdo)
# plt.xlabel('n = 1000')
# plt.yticks(np.arange(0, 151, 50))
# plt.ylim(0, 150)
# plt.ylabel('cost (MB)')
# plt.bar(xt - width, fdoA, width=width, label="Liu-FSSP (CS)", color="b")
# plt.bar(xt, fdoB, width=width, label="Liu-FSSP (ES)", color="darkblue")
# plt.bar(xt + width, qdoA, width=width, label="Ours-SPE (CS)", color="g")
# plt.bar(xt + 2*width, qdoB, width=width, label="Ours-SPE (ES)", color="darkgreen")
# plt.legend(loc=2)
# plt.savefig('D:\\fig\\CostREAL.pdf', bbox_inches='tight')
# plt.show()
