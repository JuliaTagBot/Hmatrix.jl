import re
from matplotlib import pyplot as plt

Str = r"""
LU   0.000026 seconds (6 allocations: 6.813 KiB)
L   0.054004 seconds (17.32 k allocations: 869.126 KiB)
U   0.000063 seconds (11 allocations: 5.484 KiB)
+   0.000062 seconds (21 allocations: 7.750 KiB)
LU   0.000017 seconds (6 allocations: 6.813 KiB)
L   0.063548 seconds (45.12 k allocations: 2.266 MiB)
U   0.000308 seconds (130 allocations: 77.047 KiB)
+   0.000805 seconds (508 allocations: 217.594 KiB)
LU   0.000014 seconds (6 allocations: 6.813 KiB)
L   0.000022 seconds (16 allocations: 10.016 KiB)
U   0.000021 seconds (11 allocations: 5.031 KiB)
+   0.000020 seconds (21 allocations: 6.469 KiB)
LU   0.000011 seconds (6 allocations: 6.813 KiB)
L   0.072517 seconds (77.86 k allocations: 4.127 MiB, 4.36% gc time)
U   0.006353 seconds (1.16 k allocations: 342.188 KiB)
+   0.001685 seconds (449 allocations: 396.813 KiB)
LU   0.000654 seconds (6 allocations: 25.188 KiB)
L   0.000332 seconds (16 allocations: 49.391 KiB)
U   0.000375 seconds (15 allocations: 40.938 KiB)
+   0.000062 seconds (16 allocations: 25.297 KiB)
LU   0.002415 seconds (6 allocations: 25.188 KiB)
L   0.005397 seconds (2.38 k allocations: 1.635 MiB)
U   0.005606 seconds (2.44 k allocations: 1.240 MiB)
+   0.005626 seconds (583 allocations: 1.474 MiB)
LU   0.037570 seconds (9 allocations: 97.484 KiB)
L   0.000275 seconds (22 allocations: 193.328 KiB)
U   0.000296 seconds (20 allocations: 160.703 KiB)
+   0.000240 seconds (19 allocations: 97.156 KiB)
LU   0.025005 seconds (9 allocations: 97.484 KiB)
L   0.015298 seconds (3.61 k allocations: 6.089 MiB)
U   0.014557 seconds (3.75 k allocations: 4.565 MiB)
+   0.023954 seconds (825 allocations: 5.868 MiB, 14.83% gc time)
LU   0.041692 seconds (9 allocations: 386.547 KiB)
L   0.000516 seconds (22 allocations: 769.859 KiB)
U   0.000724 seconds (20 allocations: 640.703 KiB)
+   0.000727 seconds (19 allocations: 385.156 KiB)
LU   0.061873 seconds (9 allocations: 386.547 KiB)
L   0.133387 seconds (8.46 k allocations: 33.053 MiB, 1.93% gc time)
U   0.184613 seconds (8.51 k allocations: 27.006 MiB, 0.94% gc time)
+   0.237417 seconds (5.03 k allocations: 43.152 MiB, 2.43% gc time)
LU   0.022732 seconds (9 allocations: 386.547 KiB)
L   0.000513 seconds (22 allocations: 769.859 KiB)
U   0.000741 seconds (20 allocations: 640.703 KiB)
+   0.002041 seconds (19 allocations: 385.156 KiB, 75.17% gc time)
LU   0.084357 seconds (9 allocations: 386.547 KiB)
L   0.008571 seconds (126 allocations: 3.760 MiB)
U   0.004172 seconds (122 allocations: 3.255 MiB)
+   0.003277 seconds (352 allocations: 3.522 MiB)
LU   0.028876 seconds (9 allocations: 386.547 KiB)
L   0.000619 seconds (22 allocations: 769.859 KiB)
U   0.000836 seconds (20 allocations: 640.703 KiB)
+   0.001250 seconds (19 allocations: 385.156 KiB)
LU   0.032181 seconds (9 allocations: 386.547 KiB)
"""

add = 0
L = 0
U = 0
LU = 0

for l in Str.split("\n"):
    if len(l)<10:
        continue
    
    if l[0] == "+":
        add += float(re.findall("(\d+\.\d+)", l)[0])
    elif l[0:2] == "LU":
        LU += float(re.findall("(\d+\.\d+)", l)[0])
    elif l[0]=="L":
        L += float(re.findall("(\d+\.\d+)", l)[0])
    elif l[0]=="U":
        U += float(re.findall("(\d+\.\d+)", l)[0])

plt.bar([1,2,3,4], [add, L, U, LU],tick_label=["+","LSolve","USolve","LUSolve"])
plt.show()
