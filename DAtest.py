import numpy as np
from numpy import matrix
from numpy import linalg
import DAmodel4 as DAm
import DAmodel4 as DAm2
import Data2 as D
import DACost3 as DAC3
d=D.dalecData()
import math
import matplotlib.pyplot as plt
#import DACost3 as DAC3

x0=[56.,100.,760.,35.,9888.]
dx=[5.6,10.,76.,3.5,988.8]

#Test for Linear model
def Lmodtest(lam,i):
    x0h=[x0[0]+lam*dx[0],x0[1]+lam*dx[1],x0[2]+lam*dx[2],x0[3]+lam*dx[3],x0[4]+lam*dx[4]]
    dxlam=[lam*dx[0],lam*dx[1],lam*dx[2],lam*dx[3],lam*dx[4]]
    mxdx=DAm.nldalec(x0h,i)[i-1,0]
    mx=DAm.nldalec(x0,i)[i-1,0]
    Mdx=DAm.ldalec2(x0,dxlam,i)[i-1,0]
    err=(np.linalg.norm(mxdx)-np.linalg.norm(mx))/np.linalg.norm(Mdx)
    return err
    
#Test for Linear model
def Lmodtest2(lam,i):
    x0=[56.,100.,760.,35.,9888.]
    dx=[5.6,10.,76.,3.5,988.8]
    x0h=[x0[0]+lam*dx[0],x0[1]+lam*dx[1],x0[2]+lam*dx[2],x0[3]+lam*dx[3],x0[4]+lam*dx[4]]
    dxlam=[lam*dx[0],lam*dx[1],lam*dx[2],lam*dx[3],lam*dx[4]]
    mxdx=DAm.nldalec(x0h,i)[i-1,2]
    mx=DAm.nldalec(x0,i)[i-1,2]
    Mlist=DAm.ldalec(x0,i)
    Mdx=DAC3.Mfac(Mlist,i-1)*np.matrix(dxlam).T
    err=(np.linalg.norm(mxdx)-np.linalg.norm(mx))/np.linalg.norm(Mdx[2])
    return err #(np.linalg.norm(mxdx))/(np.linalg.norm(Mdx)+np.linalg.norm(mx))
    
#Test for Linear model with C_f only argument
def Lmodtest3(lam,i):
    x0=np.array([58.,7,7,7,7])
    dx=np.array([5.8,7,7,7,7])
    x0h=x0+lam*dx
    dxlam=lam*dx
    mxdx=DAm.nldalec(x0h,i)[0][i-1]
    mx=DAm.nldalec(x0,i)[0][i-1]
    #Mlist=DAm.ldalec(x0,i)[0]
    #Mdx=DAC3.Mfac(Mlist,i)*np.matrix(dxlam).T
    Mdx=DAm.ldalec2(x0,dxlam,i)[i-1]
    err=(mxdx[0]-mx[0])/(Mdx[0])
    return err
    
def Lmodtestfunc2(lam,i):
    print Lmodtest3(lam,i)
    print Lmodtest3(lam*0.1,i)
    print Lmodtest3(lam*0.001,i)
    print Lmodtest3(lam*0.0001,i)
    print Lmodtest3(lam*0.00001,i)
    print Lmodtest3(lam*0.000001,i)

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
    x0h=np.array(np.matrix(x0).T+ hM) #x0h=[x0[0]+b,x0[1]+b,x0[2]+b,x0[3]+b,x0[4]+b]
    a=DAC3.J(x0h)-DAC3.J(x0)
    c=hM.T*DAC3.dJ(x0)
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

def Lmodtestfunc(lam,i):
    print Lmodtest2(lam,i)
    print Lmodtest2(lam*0.1,i)
    print Lmodtest2(lam*0.001,i)
    print Lmodtest2(lam*0.0001,i)
    print Lmodtest2(lam*0.00001,i)
    print Lmodtest2(lam*0.000001,i)
    
def Gradtestfunc(lam):
    print Gradtest(lam)
    print Gradtest(lam*0.1)
    print Gradtest(lam*0.001)
    print Gradtest(lam*0.0001)
    print Gradtest(lam*0.00001)
    print Gradtest(lam*0.000001)
