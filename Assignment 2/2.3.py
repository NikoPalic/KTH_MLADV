import numpy as np
import math
from scipy.stats import norm, gamma
from scipy.special import gamma as gammaf
import matplotlib.pyplot as plt
from math import sqrt, pi, e

np.random.seed(0)
N = 100
MU, SIGMA = 5, 2
PRECISION = 1./SIGMA**2
MU_0 = 100; LAMBDA_0 = 30; A_0 = 0; B_0 = 0 #prior belief
SAMPLES = np.random.normal(MU, SIGMA, N)
SAMPLE_MEAN = sum(SAMPLES)/N
SAMPLE_VAR = 0
for i in SAMPLES:
    SAMPLE_VAR+=(i-SAMPLE_MEAN)**2
SAMPLE_VAR/=N #not N-1?
#print(SAMPLES)

PLOT={"mus": (25, 30), #1, 10 #25, 30
      "taus": (0, 0.001)} #0, 0.5 #0, 0.001

def plot_true_posterior():
    def compute_normalgamma_pdf(mu, llambda, a, b):
        return lambda x, tau : b**a*sqrt(llambda)/(gammaf(a)*sqrt(2*pi))*(tau**(a-0.5))*(e**(-b*tau))*(e**-(llambda*tau*(x-mu)**2/2))
    def compute_normalgamma_pdfsample():
        mu = (LAMBDA_0*MU_0+N*SAMPLE_MEAN)/(LAMBDA_0+N)
        llambda = LAMBDA_0 + N
        a = A_0+N/2.
        b = B_0+0.5*(N*SAMPLE_VAR+(LAMBDA_0*N*(SAMPLE_MEAN-MU_0)**2)/(LAMBDA_0+N))
        print(mu, llambda, a, b)
        return compute_normalgamma_pdf(mu, llambda, a, b)
    func = compute_normalgamma_pdfsample()

    mus = np.linspace(*PLOT["mus"], 100)
    taus = np.linspace(*PLOT["taus"], 100)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = func(mus[i],taus[j])
    #print(Z)
    plt.xlabel("mu"); plt.ylabel("tau")
    plt.contour(M, T, Z, levels=10)
    plt.show()
plot_true_posterior()
##########################################################################
def get_gaussian_params(E_tau):
    mu = (LAMBDA_0*MU_0+N*SAMPLE_MEAN)/(LAMBDA_0+N)
    llambda = (LAMBDA_0+N)*E_tau
    return mu, llambda

def get_gamma_params(E_mu, E_mu2):
    a = A_0 + N/2.
    '''equation 10.30'''
    left_sum=0
    for i in range(N):
        left_sum+=SAMPLES[i]**2
        left_sum-=2*SAMPLES[i]*E_mu
        left_sum+=E_mu2
    right_item=LAMBDA_0*(E_mu2-2*MU_0*E_mu+MU_0**2)
    b=B_0+0.5*(left_sum+right_item)
    return a, b

def compute_from_gaussian(mu, llambda): #return first and second moment
    return mu, mu**2 + 1./llambda

def compute_from_gamma(a, b):
    return a/b

def pdf_gaussian(mmu, mu, llambda):
    return norm.pdf(mmu, mu, np.sqrt(1/llambda))

def pdf_gamma(tau, a, b):
    return gamma.pdf(tau, a, loc=0, scale=1/b)

E_tau = 100 #guess
for iteration in range(10):
    mu, llambda = get_gaussian_params(E_tau)
    E_mu, E_mu2 = compute_from_gaussian(mu, llambda)
    a, b = get_gamma_params(E_mu, E_mu2)
    E_tau = compute_from_gamma(a, b)
    print(mu, llambda, a, b)

    #plot
    mus = np.linspace(*PLOT["mus"], 100) #1, 10
    taus = np.linspace(*PLOT["taus"], 100) #0, 0.5
    M, T = np.meshgrid(mus, taus, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = pdf_gaussian(mus[i], mu, llambda) * pdf_gamma(taus[j], a, b)
    plt.xlabel("mu"); plt.ylabel("tau")
    plt.contour(M, T, Z, levels = 10)
    plt.show()