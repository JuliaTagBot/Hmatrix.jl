import matplotlib.pyplot as plt
import re
import numpy as np

def parse_construction_lu_matvec(file):
    R = np.zeros((5,8))
    with open(file, "r") as fp:
        cnt = fp.read()
    res = re.findall("storage=(.*?)\\n", cnt)
    res = res[::2]
    R[0,:] = list(map(float, res))

    res = re.findall("Construction : \(Hmat\)(.*?) sec", cnt)
    R[1,:] = list(map(float, res))

    res = re.findall("LU           : \(Hmat\)(.*?) sec", cnt)
    R[2,:] = list(map(float, res))

    res = re.findall("MatVec       : \(Hmat\)(.*?) sec", cnt)
    R[3,:] = list(map(float, res))

    res = re.findall("Solve        : \(Hmat\)(.*?) sec", cnt)
    R[4,:] = list(map(float, res))

    return R

def parse_construction_lu_matvec2(file):
    R = np.zeros((4,7))
    with open(file, "r") as fp:
        cnt = fp.read()
    res = re.findall("\(Full\)(.*?) sec", cnt)
    R[0,:] = list(map(float, res[::4]))
    R[1,:] = list(map(float, res[1::4]))
    R[2,:] = list(map(float, res[2::4]))
    R[3,:] = list(map(float, res[3::4]))

    return R

def compress():
    R = np.zeros((2,8))
    with open("h32_3.txt", "r") as fp:
        cnt = fp.read()
    res = re.findall("compress=(.*?),", cnt)
    res = res[::2]
    R[0,:] = list(map(float, res))

    with open("h64_3.txt", "r") as fp:
        cnt = fp.read()
    res = re.findall("compress=(.*?),", cnt)
    res = res[::2]
    R[1,:] = list(map(float, res))

    plt.title("Compress Ratio")
    plt.loglog(2**np.arange(10,18), R[0,:], "o-", label="$N_{\min}=32$")
    plt.loglog(2**np.arange(10,18), R[1,:], "o-", label="$N_{\min}=64$")
    plt.xlabel("N")
    plt.ylabel("Compress Ratio")
    plt.legend()
    plt.show()

def full_rk():
    R = np.zeros((2,8))
    with open("h64_3.txt", "r") as fp:
        cnt = fp.read()
    res = re.findall("full=(.*?),", cnt)
    res = res[::2]
    R[0,:] = list(map(float, res))
    res = re.findall("rk=(.*?),", cnt)
    res = res[::2]
    R[1,:] = list(map(float, res))
    x = 2**np.arange(10,18)
    fig, ax = plt.subplots()
    ax.stackplot(x, R[0,:], R[1,:], labels=["Full Subblocks", "Low Rank Subblocks"])
    ax.legend(loc='upper left')
    plt.xlabel("N")
    plt.ylabel("number of blocks")
    plt.show()



R1 = parse_construction_lu_matvec("h32_2.txt")
R2 = parse_construction_lu_matvec("h64_2.txt")
R3 = parse_construction_lu_matvec("h32_3.txt")
R4 = parse_construction_lu_matvec("h64_3.txt")
R5 = parse_construction_lu_matvec2("32_2.txt")

def storage():
    plt.title("Storage")
    n1 = 2**np.arange(10,18)
    n2 = 2**np.arange(10,17)
    plt.loglog(n1, R1[0,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R2[0,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R3[0,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=8$")
    plt.loglog(n1, R4[0,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=8$")
    plt.loglog(n1,1e2*n1,"--",label="$\mathcal{O}(N)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Storage (bytes)")
    plt.show()

def construction():
    plt.title("Construction")
    n1 = 2**np.arange(10,18)
    n2 = 2**np.arange(10,17)
    plt.loglog(n1, R1[1,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R2[1,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R3[1,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=8$")
    plt.loglog(n1, R4[1,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=8$")

    plt.loglog(n2, R5[0,:], "o-", label="Full Matrix")
    plt.loglog(n1,1e-4*n1,"--",label="$\mathcal{O}(N)$")
    plt.loglog(n1,1e-6*n1**2,"--",label="$\mathcal{O}(N^2)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

def LU():
    plt.title("LU")
    n1 = 2**np.arange(10,18)
    n2 = 2**np.arange(10,17)
    plt.loglog(n1, R1[2,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R2[2,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R3[2,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=8$")
    plt.loglog(n1, R4[2,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=8$")
    plt.loglog(n2, R5[2,:], "o-", label="Full Matrix")
    plt.loglog(n1,1e-5*n1,"--",label="$\mathcal{O}(N)$")
    plt.loglog(n1,5e-12*n1**3,"--",label="$\mathcal{O}(N^3)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

def MatVec():
    plt.title("MatVec")
    n1 = 2**np.arange(10,18)
    n2 = 2**np.arange(10,17)
    plt.loglog(n1, R1[3,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R2[3,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=4$")
    plt.loglog(n1, R3[3,:], "o-", label="$N_{\min}=32, N_{\mathrm{block}}=8$")
    plt.loglog(n1, R4[3,:], "o-", label="$N_{\min}=64, N_{\mathrm{block}}=8$")
    plt.loglog(n2, R5[1,:], "o-", label="Full Matrix")
    plt.loglog(n1,1e-6*n1,"--",label="$\mathcal{O}(N)$")
    plt.loglog(n1,1e-9*n1**2,"--",label="$\mathcal{O}(N^2)$")

    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

def batch_plot():
    p = """Explicit Matrix: (.*?) seconds, (.*?) bytes
Implicit Matrix: (.*?) seconds, (.*?) bytes
LU: (.*?) seconds, (.*?) bytes
Iteration: (.*?) seconds, (.*?) bytes"""
    with open("hmat.txt","r") as fp:
        cnt = fp.read()
    g = re.findall(p, cnt)
    R = np.zeros((2,7))
    for i in range(7):
        ss = g[i]
        ss = list(map(float, ss))
        R[0,i] = sum([ss[0], ss[2], ss[4], ss[6]])
        R[1,i] = sum([ss[1], ss[3]])

    with open("full.txt","r") as fp:
        cnt = fp.read()

    g = re.findall(p, cnt)
    S = np.zeros((2,7))
    for i in range(7):
        ss = g[i]
        ss = list(map(float, ss))
        S[0,i] = sum([ss[0], ss[2], ss[4], ss[6]])
        S[1,i] = sum([ss[1], ss[3]])
    
    plt.loglog(2**np.arange(10,17),R[0,:], "o-", label="$\mathcal{H}$-matrix")
    plt.plot(2**np.arange(10,17),S[0,:], "o-", label="Full matrix")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.title("Time Consumption")
    plt.show()

    plt.figure()
    plt.loglog(2**np.arange(10,17),R[1,:], "o-", label="$\mathcal{H}$-matrix")
    plt.plot(2**np.arange(10,17),S[1,:], "o-", label="Full matrix")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Storage (bytes)")
    plt.title("Storage Consumption")
    plt.show()

def D2plot():
    with open("2D.txt","r") as fp:
        cnt = fp.read()
    H = re.findall("Hmat:(\d+\.\d+)", cnt)
    D = re.findall("Full:(\d+\.\d+)", cnt)
    H = list(map(float, H))
    D = list(map(float, D))
    n = [256, 1024, 4096, 16384, 65536]
    n1 = np.array(n)
    
    plt.subplot(221)
    plt.loglog(n, H[::4] ,"o-", label="$\mathcal{H}$-matrix")
    plt.loglog(n, D[::4], "o-", label="full matrix")
    plt.loglog(n1, D[0]*(n1/256)**2,"--",label="$\mathcal{O}(N^2)$")
    plt.legend()
    plt.ylabel("Time (sec)")
    plt.title("Construction")
    plt.subplot(222)
    plt.loglog(n, H[1::4], "o-", label="$\mathcal{H}$-matrix")
    plt.loglog(n, D[1::4], "o-", label="full matrix")
    plt.loglog(n1, D[1]*(n1/256)**2,"--",label="$\mathcal{O}(N^2)$")
    plt.legend()
    plt.ylabel("Time (sec)")
    plt.title("MatVec")
    plt.subplot(223)
    plt.loglog(n, H[2::4], "o-", label="$\mathcal{H}$-matrix")
    plt.loglog(n, D[2::4], "o-", label="full matrix")
    plt.loglog(n1, D[2]*(n1/256)**3,"--",label="$\mathcal{O}(N^3)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.title("LU")
    plt.subplot(224)
    plt.loglog(n, H[3::4], "o-", label="$\mathcal{H}$-matrix")
    plt.loglog(n, D[3::4], "o-", label="full matrix")
    plt.loglog(n1, D[3]*(n1/256)**2,"--",label="$\mathcal{O}(N^2)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.title("Solve")
    plt.show()


        


