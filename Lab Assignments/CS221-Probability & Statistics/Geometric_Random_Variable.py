import matplotlib.pyplot as plt

def _pmf_geometric(x, p):
    return ((1-p)**(x-1))*p

def pdf_hist(n0, n, p):
    attempts_to_show = range(n+1)[1:]
    plt.xlabel('trials')
    plt.ylabel('probability')
    barlist = plt.bar(attempts_to_show, height=[_pmf_geometric(x, p) for x in attempts_to_show], tick_label=attempts_to_show)
    barlist[n0-1].set_color('m')
    plt.show()

def cdf_hist(k, p):
    pVal = 0
    r= list(range(k))
    dist = []
    for i in range(1, k+1):
        pmf_i = _pmf_geometric(i, p)
        pVal += pmf_i
        dist.append(pVal)
    plt.bar(r, dist)
    plt.show()

def __init__():
    n = int(input("Enter number of trials (n): "))
    p = float(input("Enter probability (p): "))
    n0 = int(input("Enter the number you want to check the pmf of in the histogram (n0): "))
    k = int(input("Enter number whose cdf you want to find (k): "))

    pdf_hist(n0, n, p)
    cdf_hist(k, p)
__init__()