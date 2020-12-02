import numpy as np
from scipy.integrate import nquad
from matplotlib import pyplot as plt


def adist(ax, w):
    return (w**2/(4*np.pi))**(0.5)*np.exp(-0.25*ax**2*w**2)

def a(ax, w, theta, phi):
    return adist(ax, w)*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def costheta(ax1, theta1, phi1, ax2, theta2, phi2, B, w):
    b0 = np.array([0, 0, B])
    a1 = a(ax1, w, theta1, phi1)
    a2 = a(ax2, w, theta2, phi2)
    v1 = b0+a1
    v2 = b0+a2
    ct = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if ct > 1:
        ct = 1
    elif ct < -1:
        ct = -1
    return ct

def sf(ax1, theta1, phi1, ax2, theta2, phi2, B, w, kS, kT):
    theta = np.arccos(costheta(ax1, theta1, phi1, ax2, theta2, phi2, B, w))
    cst2 = np.cos(theta/2)**2
    sst2 = np.sin(theta/2)**2
    return 0.5*(((kS*0.5*sst2)/(kS*0.5*sst2+kT*(0.5*sst2+cst2)))+((kS*0.5*cst2)/(kS*0.5*cst2+kT*(0.5*cst2+sst2))))

def singlet_fraction(B, w, kS, kT):
    return nquad(sf, [(-w, w), (0, np.pi), (0, 2*np.pi), (-w, w), (0, np.pi), (0, 2*np.pi)], args=[B, w, kS, kT])

if __name__ == '__main__':
    
    kS = 1
    kT = 1
    w = 50
    Bs = np.linspace(0, 300, 2)
    
    chi_s = np.zeros_like(Bs)
    for i, B in enumerate(Bs):
        chi_s[i] = singlet_fraction(B, w, kS, kT)
    MFE = (chi_s-chi_s[0])/chi_s[0]
    
    fig, ax = plt.subplots()
    ax.plot(Bs, MFE)
        