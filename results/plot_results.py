import matplotlib.pyplot as plt
import re
import numpy as np

def parse_construction_lu_matvec(file):
    R = np.zeros((4,9))
    with open(file, "r") as fp:
        cnt = fp.read()
    res = re.findall("storage=(.*?)\\n", cnt)
    R[0,:] = list(map(float, res))

    res = re.findall("Construction : \(Hmat\)(.*?) sec", cnt)
    R[1,:] = list(map(float, res))

    res = re.findall("LU           : \(Hmat\)(.*?) sec", cnt)
    R[2,:] = list(map(float, res))

    res = re.findall("MatVec       : \(Hmat\)(.*?) sec", cnt)
    R[3,:] = list(map(float, res))

    return R

def parse_construction_lu_matvec2(file):
    R = np.zeros((3,8))
    with open(file, "r") as fp:
        cnt = fp.read()
    res = re.findall("\(Full\)(.*?) sec", cnt)
    R[0,:] = list(map(float, res[::3]))
    R[1,:] = list(map(float, res[1::3]))
    R[2,:] = list(map(float, res[2::3]))

    return R

def compress():
    R = np.zeros((2,9))
    with open("32.txt", "r") as fp:
        cnt = fp.read()
    res = re.findall("compress=(.*?),", cnt)
    R[0,:] = list(map(float, res))

    with open("64.txt", "r") as fp:
        cnt = fp.read()
    res = re.findall("compress=(.*?),", cnt)
    R[1,:] = list(map(float, res))

    plt.title("Compress Ratio")
    plt.loglog(2**np.arange(9,18), R[0,:], "o-", label="$N_{\min}=32$")
    plt.loglog(2**np.arange(9,18), R[1,:], "o-", label="$N_{\min}=64$")
    plt.xlabel("N")
    plt.ylabel("Compress Ratio")
    plt.show()

def full_rk():
    R = np.zeros((2,9))
    with open("64.txt", "r") as fp:
        cnt = fp.read()
    res = re.findall("full=(.*?),", cnt)
    R[0,:] = list(map(float, res))
    res = re.findall("rk=(.*?),", cnt)
    R[1,:] = list(map(float, res))
    x = 2**np.arange(9,18)
    fig, ax = plt.subplots()
    ax.stackplot(x, R[0,:], R[1,:], labels=["Full Subblocks", "Low Rank Subblocks"])
    ax.legend(loc='upper left')
    plt.xlabel("N")
    plt.ylabel("number of blocks")
    plt.show()



R1 = parse_construction_lu_matvec("32.txt")
R2 = parse_construction_lu_matvec("64.txt")
R3 = parse_construction_lu_matvec2("full.txt")

def storage(R1, R2, R3):
    plt.title("Storage")
    plt.loglog(2**np.arange(9,18), R1[0,:], "o-", label="$N_{\min}=32$")
    plt.loglog(2**np.arange(9,18), R2[0,:], "o-", label="$N_{\min}=64$")
    plt.loglog(2**np.arange(9,18),1e3*2**np.arange(9,18),"--",label="$\mathcal{O}(N)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Storage (bytes)")
    plt.show()

def construction(R1, R2, R3):
    plt.title("Construction")
    plt.loglog(2**np.arange(9,18), R1[1,:], "o-", label="$N_{\min}=32$")
    plt.loglog(2**np.arange(9,18), R2[1,:], "o-", label="$N_{\min}=64$")
    plt.loglog(2**np.arange(9,17), R3[0,:], "o-", label="Full Matrix")
    plt.loglog(2**np.arange(9,18),1e-4*2**np.arange(9,18),"--",label="$\mathcal{O}(N)$")
    plt.loglog(2**np.arange(9,18),1e-6*(2**np.arange(9,18))**2,"--",label="$\mathcal{O}(N^2)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

def LU(R1, R2, R3):
    plt.title("LU")
    plt.loglog(2**np.arange(9,18), R1[2,:], "o-", label="$N_{\min}=32$")
    plt.loglog(2**np.arange(9,18), R2[2,:], "o-", label="$N_{\min}=64$")
    plt.loglog(2**np.arange(9,17), R3[1,:], "o-", label="Full Matrix")
    N = 2**np.arange(9,18)
    plt.loglog(2**np.arange(9,18),1e-11*(2**np.arange(9,18))**3,"--",label="$\mathcal{O}(N^3)$")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

def MatVec(R1, R2, R3):
    plt.title("MatVec")
    plt.loglog(2**np.arange(9,18), R1[3,:], "o-", label="$N_{\min}=32$")
    plt.loglog(2**np.arange(9,18), R2[3,:], "o-", label="$N_{\min}=64$")
    plt.loglog(2**np.arange(9,17), R3[2,:], "o-", label="Full Matrix")
    N = 2**np.arange(9,18)
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

def batch_plot():
    p = """Explicit Matrix: (.*?) seconds, (.*?) bytes
Implicit Matrix: (.*?) seconds, (.*?) bytes
LU: (.*?) seconds, (.*?) bytes
Iteration: (.*?) seconds, (.*?) bytes"""
    with open("batch1.txt","r") as fp:
        cnt = fp.read()
    g = re.findall(p, cnt)
    R = np.zeros((2,7))
    for i in range(7):
        ss = g[i+1]
        ss = list(map(float, ss))
        R[0,i] = sum([ss[0], ss[2], ss[4], ss[6]])
        R[1,i] = sum([ss[1], ss[3]])

    with open("batch2.txt","r") as fp:
        cnt = fp.read()

    g = re.findall(p, cnt)
    S = np.zeros((2,7))
    for i in range(7):
        ss = g[i+1]
        ss = list(map(float, ss))
        S[0,i] = sum([ss[0], ss[2], ss[4], ss[6]])
        S[1,i] = sum([ss[1], ss[3]])
    
    plt.plot(2**np.arange(10,17),R[0,:], "o-", label="$\mathcal{H}$-matrix")
    plt.plot(2**np.arange(10,17),S[0,:], "o-", label="Full matrix")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.show()

    plt.figure()
    plt.plot(2**np.arange(10,17),R[1,:], "o-", label="$\mathcal{H}$-matrix")
    plt.plot(2**np.arange(10,17),S[1,:], "o-", label="Full matrix")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Storage (bytes)")
    plt.show()
        


