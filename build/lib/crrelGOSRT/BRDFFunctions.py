
import numpy as np

## FOR debug, importm matplotlib
from matplotlib import pyplot as plt

def BRDFProb(nPhotons,Zen,azi_i,phi,theta,dTheta,ParamDict,BRDF='Hapke'):

    justPhi=phi[:]
    justTheta=theta[:]
    theta,phi=np.meshgrid(theta,phi)

    if BRDF.lower() == 'hapke':
        Rho=HapkeBRDF(Zen,azi_i,theta,phi,omega=ParamDict['omega'],b_0=ParamDict['b_0'],hh=ParamDict['hh'])
    elif BRDF.lower() == 'rpv':
        Rho=RVPBRDF(Zen,azi_i,theta,phi,rho_0=ParamDict['rho_0'],k=ParamDict['k'],g=ParamDict['g'],h_0=ParamDict['h_0'])
    else:
        Rho=HapkeBRDF(Zen,azi_i,theta,phi,omega=ParamDict['omega'],b_0=ParamDict['b_0'],hh=ParamDict['hh'])

    HDRF=np.sum(Rho*np.cos(theta)*np.sin(theta)*dTheta*dTheta)

    if np.max(HDRF) > 1.:
        HDRF=0.9
        # print(np.max(HDRF),np.degrees(Zen),np.degrees(azi_i))
        # print("This HDRF (reflectance) is greater than 1, not physically possible, exiting.")
        #
        # plt.figure()
        # ax=plt.subplot(111,projection='polar')
        # Z=plt.pcolormesh(phi,np.degrees(theta),(Rho*np.cos(theta)*np.sin(theta)*dTheta*dTheta)/HDRF,cmap='jet')
        # plt.colorbar(Z)
        # plt.show()
        #
        # sys.exit()

    prob=((Rho*np.cos(theta)*np.sin(theta)*dTheta*dTheta)/HDRF).ravel()

    ThetaProb=theta.ravel()
    PhiProb=phi.ravel()

    RandThetas=np.random.choice(ThetaProb,nPhotons,p=prob/np.sum(prob))
    RandPhis=np.random.choice(PhiProb,nPhotons,p=prob/np.sum(prob))

    rands1=np.random.uniform(-1,1.,size=nPhotons)*dTheta
    rands2=np.random.uniform(-1.,1.,size=nPhotons)*dTheta

    RandThetas+=rands1
    RandPhis+=rands2

    return HDRF, RandPhis,RandThetas

def RVPBRDF(zen_i,azi_i,zen_r,azi_r,rho_0=0.027,k=0.647,g=-0.169,h_0=0.1):

    mu_i=np.cos(zen_i)
    mu_r=np.cos(zen_r)
    dAzi=azi_i-azi_r

    cosScatter=mu_i*mu_r + np.sqrt(1.-mu_i**2.)*np.sqrt(1.-mu_r**2.)*np.cos(dAzi)
    alpha=np.pi-np.arccos(cosScatter)

    Term1=rho_0*(mu_i*mu_r*(mu_i+mu_r))**(k-1.)
    FHG=(1-g**2.)/((1+g**2.-2*g*np.cos(np.pi-alpha))**(3./2.))

    G=np.sqrt(np.tan(np.arccos(mu_i))**2.+np.tan(np.arccos(mu_r))**2. -
                2. * np.tan(np.arccos(mu_r))*np.tan(np.arccos(mu_i))*np.cos(dAzi))

    h=1.+(1-h_0)/(1.+G)

    Rho=Term1*h*FHG

    return Rho

def HapkeBRDF(zen_i,azi_i,zen_r,azi_r,omega=0.6,b_0=1.,hh=0.06):
    mu_i=np.cos(zen_i)
    mu_r=np.cos(zen_r)
    dAzi=azi_i-azi_r

    cosScatter=mu_i*mu_r + np.sqrt(1.-mu_i**2.)*np.sqrt(1.-mu_r**2.)*np.cos(np.pi-dAzi)
    Pcos=1.+cosScatter/2.

    h0=(1.+2.*mu_i)/(1.+2.*mu_i*np.sqrt(1.-omega))
    h=(1.+2.*mu_r)/(1.+2.*mu_r*np.sqrt(1.-omega))
    b=(b_0*hh)/(hh+np.tan(np.arccos(cosScatter)/2.))

    Rho=omega/4.*(((1.+b)*Pcos+h0*h-1.)/(mu_i+mu_r))

    return Rho
