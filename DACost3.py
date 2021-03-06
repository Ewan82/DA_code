import numpy as np
from numpy import matrix
from numpy import linalg
import DAmodel4 as DAm
import Data2 as D
import math
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_ncg
d=D.dalecData()

B=matrix([[d.sigB_cf,0,0,0,0],[0,d.sigB_cw,0,0,0],[0,0,d.sigB_cr,0,0],[0,0,0,d.sigB_cl,0],[0,0,0,0,d.sigB_cs]]) #Background error covariance matrix
XB=np.matrix([58.,102.,770.,40.,9897.]).T #Background state
lenrun=365 #assimilation window length

#Matrix factoral function
def Mfac(Mlist,a): #Matrix factoral to find product of M matrices
    if a==1:
        return Mlist[0]
    else:
        Mat=Mlist[0]
        for x in xrange(0,a-1):
            Mat=Mlist[x+1]*Mat
    return Mat

#Adjoint Matrix factoral function
def MfacAdj(Mlist,a): #Matrix factoral to find product of M matrices
    if a==1:
        return Mlist[0].T
    else:
        Mat=Mlist[0].T
        for x in xrange(0,a-1):
            Mat=Mat*Mlist[x+1].T
    return Mat

#Creates size x size R matrix
def Rmat(y1_err):
    if len(y1_err)==1:
        return y1_err[0]
    matrix = [[0]*len(y1_err) for i in range(len(y1_err))]
    for i in range(len(y1_err)):
        matrix[i][i] = float(y1_err[i]) #uses elements of y_err as stnd devs
    return np.matrix(matrix)

#Cost function that takes initial state as argument
def J(X):
    nlhx=DAm.nldalec(X,lenrun)[0] #Run nonlinear model for length run outputing models obs guess
    X1=[float(X[0]),float(X[1]),float(X[2]),float(X[3]),float(X[4])]
    X0=np.matrix(X1).T

    #Creates H_i's, y_i's and y_err_i's
    hx=[] #h(x)
    y=[] #obs
    y_err=[] #obs error
    for x in range(lenrun):
        temp=[]
        yi=[]
        yi_err=[]
        obs=False
        if d.gpp[x]!=None:
            temp.append(nlhx[x,0])
            yi=np.append(yi,[d.gpp[x]])
            yi_err=np.append(yi_err,[d.gpp_err[x]**2])
            obs=True
        if d.lf[x]!=None:
            temp.append(nlhx[x,1])
            yi=np.append(yi,[d.lf[x]])
            yi_err=np.append(yi_err,[d.lf_err[x]**2])
            obs=True
        if d.lw[x]!=None:
            temp.append(nlhx[x,2])
            yi=np.append(yi,[d.lw[x]])
            yi_err=np.append(yi_err,[d.lw_err[x]**2])
            obs=True
        if d.rt[x]!=None:
            temp.append(nlhx[x,3])
            yi=np.append(yi,[d.rt[x]])
            yi_err=np.append(yi_err,[d.rt_err[x]**2])
            obs=True
        if d.nee[x]!=None:
            temp.append(nlhx[x,4])
            yi=np.append(yi,[d.nee[x]])
            yi_err=np.append(yi_err,[d.nee_err[x]**2])
            obs=True
        if d.cf[x]!=None:
            temp.append(nlhx[x,5])
            yi=np.append(yi,[d.cf[x]])
            yi_err=np.append(yi_err,[d.cf_err[x]**2])
            obs=True
        if obs==False:
            temp.append(0)
            yi=np.append(yi,0)
            yi_err=np.append(yi_err,0)
        hx.append(np.matrix(temp))
        y.append(np.matrix(yi))
        y_err.append((yi_err))

    #Create y-hx increments
    yhx=[y[i].T-hx[i].T for i in range(len(y))]
    
    Incr=[0]*lenrun
    for i in range(lenrun):
        if np.linalg.norm(y_err[i])!=0:
            Incr[i]=yhx[i].T*np.matrix(Rmat(y_err[i])).I*yhx[i]
    
    #Obs elements to be summed
    #Incr=[yhx[i].T*np.matrix(Rmat(y_err[i])).I*yhx[i] for i in range(len(y))]    

    #Cost fn
    J=0.5*(X0-XB).T*B.I*(X0-XB) +0.5*np.sum(Incr)
    
    return float(J)

#Derivative of costfunction takes initial state as argument
def dJ(X):
    Mlist,GPP_diff,M=DAm.ldalec(X,lenrun) #run linear model out list of linear model matrices and value for GPP derivative at each time step
    nlhx=DAm.nldalec(X,lenrun)[0] #run nonlinear model for obs guess output
    X1=[X[0],X[1],X[2],X[3],X[4]]
    X0=np.matrix(X1).T

    #Creates H_i's, y_i's and y_err_i's
    H=[] #linear obs operator
    hx=[] #h(x)
    y=[] #obs
    y_err=[] #obs error
    for x in range(lenrun):
        temp=[]
        temp2=[]
        yi=[]
        yi_err=[]
        obs=False
        if d.gpp[x]!=None:
            temp.append([GPP_diff[x],0,0,0,0])
            temp2.append(nlhx[x,0])
            yi=np.append(yi,[d.gpp[x]])
            yi_err=np.append(yi_err,[d.gpp_err[x]**2])
            obs=True
        if d.lf[x]!=None:
            temp.append([d.p_5,0,0,0,0])
            temp2.append(nlhx[x,1])
            yi=np.append(yi,[d.lf[x]])
            yi_err=np.append(yi_err,[d.lf_err[x]**2])
            obs=True
        if d.lw[x]!=None:
            temp.append([0,0,d.p_6,0,0])
            temp2.append(nlhx[x,2])
            yi=np.append(yi,[d.lw[x]])
            yi_err=np.append(yi_err,[d.lw_err[x]**2])
            obs=True
        if d.rt[x]!=None:
            temp.append([d.p_2*GPP_diff[x],0,0,d.p_8*d.T[x],d.p_9*d.T[x]])
            temp2.append(nlhx[x,3])
            yi=np.append(yi,[d.rt[x]])
            yi_err=np.append(yi_err,[d.rt_err[x]**2])
            obs=True
        if d.nee[x]!=None:
            temp.append([-(1-d.p_2)*GPP_diff[x],0,0,-d.p_8*d.T[x],-d.p_9*d.T[x]])
            temp2.append(nlhx[x,4])
            yi=np.append(yi,[d.nee[x]])
            yi_err=np.append(yi_err,[d.nee_err[x]**2])
            obs=True
        if d.cf[x]!=None:
            temp.append([1,0,0,0,0])
            temp2.append(nlhx[x,5])
            yi=np.append(yi,[d.cf[x]])
            yi_err=np.append(yi_err,[d.cf_err[x]**2])
            obs=True
        if obs==False:
            temp.append([0,0,0,0,0])
            temp2.append(0)
            yi=np.append(yi,0)
            yi_err=np.append(yi_err,0)
        H.append(np.matrix(np.vstack(temp)))    
        hx.append(np.matrix(temp2))
        y.append(np.matrix(yi))
        y_err.append((yi_err))

    yhx=[y[i].T-hx[i].T for i in range(len(y))] #Create y-h(x) incriment

    #stacklist1=[H[0]] #Creates H hat matrix
    #for x in xrange(1,len(y)):
    #    stacklist1.append(H[x]*Mfac(Mlist,x))

    stacklist2=[H[0].T] #creats M^T*H^T for each time step
    for x in xrange(1,len(y)):
        stacklist2.append(MfacAdj(Mlist,x)*H[x].T)

    #Incr=[stacklist2[i]*np.matrix(Rmat(y_err[i])).I*yhx[i] for i in range(len(y))] #Creates obs incriment
    
    Incr=[]
    for i in range(lenrun):
        if np.linalg.norm(y_err[i])!=0:
            Incr.append(stacklist2[i]*np.matrix(Rmat(y_err[i])).I*yhx[i])
    
    dJ=B.I*(X0-XB) -np.sum(Incr, axis=0) #Cost function first derivative
    dJlist=[float(dJ[0]),float(dJ[1]),float(dJ[2]),float(dJ[3]),float(dJ[4])]
    
    return dJ.T #np.array(dJlist) 


#2nd derivative of costfunction takes initial state as argument
def d2J(X):
    Mlist,GPP_diff=DAm.ldalec(X,lenrun)
    nlhx=DAm.nldalec(X,lenrun)
    X1=[X[0],X[1],X[2],X[3],X[4]]
    X0=np.matrix(X1).T

    #Figure out Hi's, y and y_err
    H=[]
    hx=[]
    y=[]
    y_err=[]
    for x in range(lenrun):
        temp=[]
        temp2=[]
        yi=[]
        yi_err=[]
        obs=False
        if d.gpp[x]!=None:
            temp.append([GPP_diff[x],0,0,0,0])
            temp2.append(nlhx[x,0])
            yi=np.append(yi,[d.gpp[x]])
            yi_err=np.append(yi_err,[d.gpp_err[x]**2])
	    obs=True
        if d.lf[x]!=None:
            temp.append([d.p_5,0,0,0,0])
            temp2.append(nlhx[x,1])
            yi=np.append(yi,[d.lf[x]])
            yi_err=np.append(yi_err,[d.lf_err[x]**2])
	    obs=True
        if d.lw[x]!=None:
            temp.append([0,0,d.p_6,0,0])
            temp2.append(nlhx[x,2])
            yi=np.append(yi,[d.lw[x]])
            yi_err=np.append(yi_err,[d.lw_err[x]**2])
	    obs=True
        if d.rt[x]!=None:
            temp.append([d.p_2*GPP_diff[x],0,0,d.p_8*d.T[x],d.p_9*d.T[x]])
            temp2.append(nlhx[x,3])
            yi=np.append(yi,[d.rt[x]])
            yi_err=np.append(yi_err,[d.rt_err[x]**2])
	    obs=True
        if d.nee[x]!=None:
            temp.append([-(1-d.p_2)*GPP_diff[x],0,0,-d.p_8*d.T[x],-d.p_9*d.T[x]])
            temp2.append(nlhx[x,4])
            yi=np.append(yi,[d.nee[x]])
            yi_err=np.append(yi_err,[d.nee_err[x]**2])
	    obs=True
        if obs==False:
            temp.append([0,0,0,0,0])
            temp2.append(0)
            yi=np.append(yi,0)
            yi_err=np.append(yi_err,0)
        H.append(np.matrix(np.vstack(temp)))    
        hx.append(temp2)
        y.append(yi)
        y_err.append(yi_err)

    #yhx=np.matrix((np.hstack(y)-np.hstack(hx))).T
    #R=Rmat(np.hstack(y_err))

    stacklist=[H[0]] #Creates H hat matrix
    for x in xrange(1,lenrun):
        stacklist.append(H[x]*Mfac(Mlist,x))

    #Hhat=np.matrix(np.vstack(stacklist))
    
    #d2J=B.I+Hhat.T*R.I*Hhat
    #Incr=[np.matrix(Rmat(y_err[i])).I*yhx[i] for i in range(0,lenrun,1)]
    d2jlist=[0]*lenrun
    for i in range(lenrun):
        if np.linalg.norm(y_err[i])!=0:
            d2jlist[i]=stacklist[i].T*np.matrix(Rmat(y_err[i])).I*stacklist[i]

    d2J=B.I+sum(d2jlist)
    
    return d2J

def Plot(X):
    nlhx=DAm.nldalec(X,lenrun)
    gppB=nlhx[:,0]
    gppO=d.gpp[0:lenrun]
    xlist=np.arange(0,lenrun,1)
    #xa=fmin_bfgs(J, X, dJ2, maxiter=4000)
    xa=fmin_ncg(J, X, dJ2, fhess=d2J2)
    nlhx2=DAm.nldalec(xa, lenrun)
    gppA=nlhx2[:,0]
    print xa
    plt.plot(xlist,gppO,label='GPP_O')
    plt.plot(xlist,gppB,label='GPP_B')
    plt.plot(xlist,gppA,label='GPP_A')
    plt.legend()
    plt.show()

def Plot2(X):
    nlhx=DAm.nldalec(X,lenrun)
    neeB=nlhx[:,4]
    neeO=d.nee[0:lenrun]
    #neeO2=[None]*lenrun
    #for i in range(lenrun):
    #    if neeO[i]!=None:
    #        neeO2[i]=-neeO[i]
    xlist=np.arange(0,lenrun,1)
    #xa=fmin_bfgs(J, X, dJ, maxiter=2000)
    xa=fmin_ncg(J, X, dJ2, fhess=d2J2)
    nlhx2=DAm.nldalec(xa, lenrun)
    neeA=nlhx2[:,4]
    print xa
    plt.plot(xlist,neeO,label='NEE_O')
    plt.plot(xlist,neeB,label='NEE_B')
    plt.plot(xlist,neeA,label='NEE_A')
    plt.legend()
    plt.show()

def Plot3(X):
    nlhx=DAm.nldalec(X,lenrun)
    neeB=nlhx[:,1]
    neeO=d.lf[0:lenrun]
    xlist=np.arange(0,lenrun,1)
    xa=fmin_bfgs(J, X)
    nlhx2=DAm.nldalec(xa, lenrun)
    neeA=nlhx2[:,1]
    print xa
    plt.plot(xlist,neeO,label='lf_O')
    plt.plot(xlist,neeB,label='lf_B')
    plt.plot(xlist,neeA,label='lf_A')
    plt.legend()
    plt.show()

def Plot4(X):
    nlhx=DAm.nldalec(X,lenrun)
    neeB=nlhx[:,3]
    neeO=d.rt[0:lenrun]
    xlist=np.arange(0,lenrun,1)
    xa=fmin_ncg(J, X, dJ2, fhess=d2J2)
    nlhx2=DAm.nldalec(xa, lenrun)
    neeA=nlhx2[:,3]
    print xa
    plt.plot(xlist,neeO,label='rt_O')
    plt.plot(xlist,neeB,label='rt_B')
    plt.plot(xlist,neeA,label='rt_A')
    plt.legend()
    plt.show()

def Plot5(X):
    nlhx=DAm.nldalec(X,lenrun)[0]
    gppB=nlhx[:,0]
    neeB=nlhx[:,4]
    lfB=nlhx[:,1]
    rtB=nlhx[:,3]
    #rtO=d.rt[0:lenrun]
    #gppO=d.gpp[0:lenrun]
    #neeO=d.nee[0:lenrun]
    #lfO=d.lf[0:lenrun]
    xlist=np.arange(0,lenrun,1)
    xa=fmin_ncg(J, X, dJ) #, fhess=d2J)
    #xa=fmin_bfgs(J, X) #fprime=dJ)
    nlhx2=DAm.nldalec(xa, lenrun)[0]
    rtA=nlhx2[:,3]
    gppA=nlhx2[:,0]
    neeA=nlhx2[:,4]
    lfA=nlhx2[:,1]
    gpp_err=[]
    gppO=[]
    gppday=[]
    for i in range(lenrun):
        if d.gpp_err[i]!=None:
            gpp_err.append(d.gpp_err[i])
            gppO.append(d.gpp[i])
            gppday.append(float(d.D[i]))
    nee_err=[]
    neeO=[]
    needay=[]
    for i in range(lenrun):
        if d.nee_err[i]!=None:
            nee_err.append(d.nee_err[i])
            neeO.append(d.nee[i])
            needay.append(float(d.D[i]))
    rt_err=[]
    rtO=[]
    rtday=[]
    for i in range(lenrun):
        if d.rt_err[i]!=None:
            rt_err.append(d.rt_err[i])
            rtO.append(d.rt[i])
            rtday.append(float(d.D[i]))
    lf_err=[]
    lfO=[]
    lfday=[]
    for i in range(lenrun):
        if d.lf_err[i]!=None:
            lf_err.append(d.lf_err[i])
            lfO.append(d.lf[i])
            lfday.append(float(d.D[i]))

    print xa
    plt.subplot(2,2,1)

    plt.plot(xlist,gppB,label='GPP_B')
    plt.plot(xlist,gppA,label='GPP_A')    
    plt.errorbar(gppday,gppO,yerr=gpp_err, fmt='o', label='GPP_O')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('GPPvalue (gCm$^{-2}$)')
    plt.title('DA GPP')
    plt.subplot(2,2,2)
    plt.plot(xlist,neeB,label='NEE_B')
    plt.plot(xlist,neeA,label='NEE_A')    
    plt.errorbar(needay,neeO,yerr=nee_err, fmt='o',label='NEE_O')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('NEEvalue (gCm$^{-2}$)')
    plt.title('DA NEE')
    plt.subplot(2,2,3)

    plt.plot(xlist,rtB,label='rt_B')
    plt.plot(xlist,rtA,label='rt_A')    
    plt.errorbar(rtday,rtO,yerr=rt_err, fmt='o',label='RT_O')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('RTvalue (gCm$^{-2}$)')
    plt.title('DA RT')
    plt.subplot(2,2,4)
 
    plt.plot(xlist,lfB,label='lf_B')
    plt.plot(xlist,lfA,label='lf_A')  
    plt.errorbar(lfday,lfO,yerr=lf_err, fmt='o',label='LF_O')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('LFvalue (gCm$^{-2}$)')
    plt.title('DA LF')
    plt.suptitle('DA Run with initial guess x0=%s'%(X))
    plt.show()
