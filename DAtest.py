import numpy as np
from numpy import matrix
from numpy import linalg
import DAmodel4 as DAm
reload(DAm)
import DAmodel4 as DAm2
import Data2 as D
reload(D)
import DACost3 as DAC3
reload(DAC3)
d=D.dalecData()
import math
import matplotlib.pyplot as plt

x0=[58.,102.,770.,40.,9897.]

#Test for Linear model
def LmodtestM(lam,i, p):
    x0=np.array([56.,100.,760.,35.,9888.])
    dx=np.array([5.6,10.,76.,3.5,988.8])
    x0h=x0+lam*dx
    dxlam=lam*dx
    mxdx=DAm.nldalec(x0h,i)[1][i-1]
    mx=DAm.nldalec(x0,i)[1][i-1]
    M=DAm.ldalec(x0,i)[2]
    Mdx=M*np.matrix(dxlam).T
    if p=='all':
        err=(np.linalg.norm(mxdx-mx))/(np.linalg.norm(Mdx))
    else:
        err=(np.linalg.norm(mxdx[p]-mx[p]))/(np.linalg.norm(Mdx[p]))
    return err #(np.linalg.norm(mxdx))/(np.linalg.norm(Mdx)+np.linalg.norm(mx))
    
#Test for Linear model with C_f only argument
def Lmodtestm(lam,i, p):
    x0=np.array([56.,100.,760.,35.,9888.])
    dx=np.array([5.6,10.,76.,3.5,988.8])
    dxlam=lam*dx
    x0h=x0+dxlam
    mxdx=DAm.nldalec(x0h,i)[1][i-1]
    mx=DAm.nldalec(x0,i)[1][i-1]
    #Mlist=DAm.ldalec(x0,i)[0]
    #Mdx=DAC3.Mfac(Mlist,i)*np.matrix(dxlam).T
    Mdx=DAm.ldalec2(x0,dxlam,i)[i-1]
    if p=='all':
        err=(np.linalg.norm(mxdx-mx))/(np.linalg.norm(Mdx))
    else:
        err=(np.linalg.norm(mxdx[p]-mx[p]))/(np.linalg.norm(Mdx[p]))
    return err
    
#Test for Linear model with C_f only argument
def LmodtestACM(lam,i):
    x0=np.array([5.])
    dx=np.array([0.5])
    x0h=x0+lam*dx
    dxlam=lam*dx
    mxdx=DAm.ACM(x0h,i)[0]
    mx=DAm.ACM(x0,i)[0]
    #Mlist=DAm.ldalec(x0,i)[0]
    #Mdx=DAC3.Mfac(Mlist,i)*np.matrix(dxlam).T
    M=DAm.ACM(x0,i)[1]
    err=(mxdx-mx)/(M*dxlam)
    return err
    
def Lmodtestfunc2(lam,i, p):
    a=[abs(float(Lmodtestm(lam*10**(-x),i,p))-1) for x in range (0,5)]

    plt.plot(np.arange(0,len(a),1),a)
    plt.title('Test of linear model')
    plt.ylabel(r'$ E_R = \frac{||m({\bf x}+\gamma\delta {\bf x})-m({\bf x})||}{||{\bf M}({\bf x})\gamma\delta {\bf x}||}-1 $', fontsize=20)
    plt.xlabel('Iteration')  
    plt.show()
    return a
    
#Test for Adjoint    
def Adjtest(i):
    Mlist,gppdiff=DAm.ldalec(x0,i)
    M=DAC3.Mfac(Mlist,i)
    MT=DAC3.MfacAdj(Mlist,i)
    t=M*np.matrix(dx).T
    return t.T*t, np.matrix(dx)*MT*t

#Test for gradient of cost function
def Gradtest(alph):
    h=DAC3.dJ(x0)*(1/np.linalg.norm(DAC3.dJ(x0)))
    #d=1/math.sqrt(5.)
    #b=alph*d
    #h=[d,d,d,d,d]
    hM=alph*h
    x0h=np.array(np.matrix(x0)+ hM).T #x0h=[x0[0]+b,x0[1]+b,x0[2]+b,x0[3]+b,x0[4]+b]
    a=DAC3.J(x0h)-DAC3.J(x0)
    c=hM*DAC3.dJ(x0).T
    return a/c

def Gradtest2(alph):
    h=DAC3.dJ(x0)*(1/np.linalg.norm(DAC3.dJ(x0)))
    #d=1/math.sqrt(5.)
    #b=alph*d
    #h=[d,d,d,d,d]
    hM=alph*h
    x0h=np.array(np.matrix(x0).T+ hM) #x0h=[x0[0]+b,x0[1]+b,x0[2]+b,x0[3]+b,x0[4]+b]
    a=DAC3.J(x0h)
    c=hM.T*DAC3.dJ(x0)+DAC3.J(x0)
    return a/c

def Gradtestfunc(lam):
    a=[float(Gradtest(lam*10**(-x))) for x in range (0,5)]
    plt.plot(np.arange(0,len(a),1),a)
    plt.title('Gradient test')
    plt.ylabel(r'$\frac{J({\bf x}+\alpha{\bf h})-J({\bf x})}{\alpha{\bf h}^{T}\bigtriangledown J({\bf x})} $', fontsize=20)
    plt.xlabel('Iteration') 
    plt.show()
    return a

