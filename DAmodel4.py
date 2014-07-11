import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import Data2 as D
d=D.dalecData()

#Agregated Canopy Model for GPP
def ACM(Cf,i):
	
    q=d.a_3-d.a_4
    #L=(Cf/111.)
    #e_0=((d.a_7*L**2.)/(L**2.+d.a_9))
    g_c=(((np.abs(float(d.phi_d[i])))**(d.a_10))/(0.5*float(d.T_range[i])+d.a_6*float(d.R_tot[i])))
    #p=np.exp(d.a_8*float(d.T_max[i]))*((d.a_1*d.N*L)/g_c)
    #C_i=(0.5*(d.d.C_a+q-p+np.np.sqrt((d.d.C_a+q-p)**2.-4.*(d.d.C_a*q-p*d.a_3))))
    delta=(-0.408*np.cos(((360.*(float(d.D[i])+10.)*np.pi)/(365.*180.))))
    s=(24.*np.arccos((-np.tan(d.bigdelta)*np.tan(delta)))/np.pi)
    psi=np.exp(d.a_8*float(d.T_max[i]))*((d.a_1*d.N)/(g_c*111.))
    a=np.exp(d.a_8*float(d.T_max[i]))*((d.a_1*d.N)/(g_c*111.))
    phi=d.a_7*float(d.I[i])*g_c
    I=float(d.I[i])
	
    def GPPfunc(Cf):
        GPP=.5*phi*Cf**2*(d.C_a-q+psi*Cf-np.sqrt((d.C_a+q-psi*Cf)**2-4.*(d.C_a*q-d.a_3*psi*Cf)))*(d.a_2*s+d.a_5)/(I*d.a_7*Cf**2+.5*g_c*(Cf**2+d.a_9*111.**2)*(d.C_a-q+psi*Cf-np.sqrt((d.C_a+q-psi*Cf)**2-4.*(d.C_a*q-d.a_3*psi*Cf))))        
        return GPP 

    def GPPdifffunc(Cf):
        GPP_diff=(((2.*(d.a_2*s+d.a_5))*(-49284.*q*d.a_9*psi*Cf+49284.*psi*Cf*d.a_3*d.a_9-49284.*d.C_a*d.a_9*q+24642.*psi**2*Cf**2*d.a_9+24642.*d.a_9*d.C_a**2+24642.*d.a_9*q**2)*phi*Cf*g_c+(2.*I)*(d.a_2*s+d.a_5)*psi*Cf**4*d.a_7*phi)*np.sqrt(psi**2*Cf**2+(-2.*d.C_a-2.*q+4.*d.a_3)*psi*Cf+d.C_a**2-2.*d.C_a*q+q**2)+(-(49284.*(d.a_2*s+d.a_5))*psi**3*d.a_9*phi*Cf**4+(2.*(d.a_2*s+d.a_5))*(24642.*d.C_a-98568.*d.a_3+73926.*q)*phi*psi**2*d.a_9*Cf**3+(2.*(d.a_2*s+d.a_5))*(-98568.*d.a_3*d.C_a+98568.*d.a_3*q+49284.*d.C_a*q-73926.*q**2+24642.*d.C_a**2)*phi*psi*d.a_9*Cf**2+(2.*(d.a_2*s+d.a_5))*(-73926.*d.C_a*q**2-24642.*d.C_a**3+24642.*q**3+73926.*d.C_a**2*q)*phi*d.a_9*Cf)*g_c-(2.*I)*(d.a_2*s+d.a_5)*psi**2*d.a_7*phi*Cf**5+(2.*(d.a_2*s+d.a_5))*((1.*I)*q*d.a_7+(1.*I)*d.C_a*d.a_7-(2.*I)*d.a_3*d.a_7)*phi*psi*Cf**4)/(np.sqrt(psi**2*Cf**2+(-2.*d.C_a-2.*q+4.*d.a_3)*psi*Cf+d.C_a**2-2.*d.C_a*q+q**2)*((2.*I)*d.a_7*Cf**2+g_c*Cf**2*d.C_a-1.*g_c*Cf**2*q+g_c*Cf**3*psi-1.*g_c*Cf**2*np.sqrt(psi**2*Cf**2+(-2.*d.C_a-2.*q+4.*d.a_3)*psi*Cf+d.C_a**2-2.*d.C_a*q+q**2)+12321.*g_c*d.a_9*d.C_a-12321.*g_c*d.a_9*q+12321.*g_c*d.a_9*psi*Cf-12321.*g_c*d.a_9*np.sqrt(psi**2*Cf**2+(-2.*d.C_a-2.*q+4.*d.a_3)*psi*Cf+d.C_a**2-2.*d.C_a*q+q**2))**2)    
        return GPP_diff
		
    GPP = GPPfunc(Cf)
    GPP_diff=  GPPdifffunc(Cf)
    GPP_diff2 = misc.derivative(GPPfunc,Cf) 
    return GPP, GPP_diff #, GPP_diff2

	
#Non-linear DALEC model
def nldalec(X,i):

    C_f,C_r,C_w,C_l,C_s=[float(X[0])],[float(X[1])],[float(X[2])],[float(X[3])],[float(X[4])]
    Xlist=np.array([[C_f[0],C_r[0],C_w[0],C_l[0],C_s[0]]])
    GPP,nee,rt,lf,lw,Obslist=[None]*i,[None]*i,[None]*i,[None]*i,[None]*i,[None]*i
    
    for x in xrange(1,i):

        C_f.append((1-d.p_5)*C_f[x-1]+d.p_3*(1-d.p_2)*ACM(C_f[x-1], x-1)[0])
        C_r.append((1-d.p_7)*C_r[x-1]+d.p_4*(1-d.p_3)*(1-d.p_2)*ACM(C_f[x-1], x-1)[0])
        C_w.append((1-d.p_6)*C_w[x-1]+(1-d.p_4)*(1-d.p_3)*(1-d.p_2)*ACM(C_f[x-1],x-1)[0])     
        C_l.append((1-(d.p_1+d.p_8)*d.T[x-1])*C_l[x-1]+d.p_5*C_f[x-1]+d.p_7*C_r[x-1])
        C_s.append((1-d.p_9*d.T[x-1])*C_s[x-1]+d.p_6*C_w[x-1]+d.p_1*d.T[x-1]*C_r[x-1])
        Xlist=np.append(Xlist, np.array([[C_f[x],C_r[x],C_w[x],C_l[x],C_s[x]]]), axis=0)
        
    for x in xrange(0,i):
        GPP[x]=ACM(C_f[x], x)[0]
        nee[x]=(-(1-d.p_2)*ACM(C_f[x], x)[0]+d.p_8*d.T[x]*C_l[x]+d.p_9*d.T[x]*C_s[x])
        rt[x]=(d.p_2*GPP[x]+d.p_8*d.T[x]*C_l[x]+d.p_9*d.T[x]*C_s[x])
        lf[x]=d.p_5*C_f[x]
        lw[x]=d.p_6*C_w[x]
        Obslist[x]=GPP[x],lf[x],lw[x],rt[x],nee[x],C_f[x]
          
    return np.array(Obslist) ,Xlist, nee


#Linear DALEC model !!TEST MODEL!!
def ldalec(X, i):   

    C_f=[X[0]]
    Mlist=[None]*(i)
    GPP_diff=[None]*i

    for x in xrange(0,i):
        GPP_diff[x]=ACM(C_f[x], x)[1]
        C_f.append((1-d.p_5)*C_f[x]+d.p_3*(1-d.p_2)*ACM(C_f[x], x)[0])
        Mlist[x]=(np.matrix([[(1-d.p_5)+d.p_3*(1-d.p_2)*ACM(C_f[x], x)[1],0,0,0,0],[(d.p_4*(1-d.p_3)*(1-d.p_2))*ACM(C_f[x], x)[1],(1-d.p_7),0,0,0],[(1-d.p_4)*(1-d.p_3)*(1-d.p_2)*ACM(C_f[x], x)[1],0,(1-d.p_6),0,0],[d.p_5,d.p_7,0,1-(d.p_1+d.p_8)*d.T[x],0],[0,0,d.p_6,d.p_1*d.T[x],1-d.p_9*d.T[x]]]))
    
    M=Mlist[0]    
    for x in xrange(1,i-1):
        M=Mlist[x]*M

    return Mlist, GPP_diff, M
    
    
#Linear DALEC model !!TEST MODEL!!
def ldalec2(X, dX, i):   

    C_f=[X[0]]
    C_fL,C_rL,C_wL,C_lL,C_sL=[dX[0]],[dX[1]],[dX[2]],[dX[3]],[dX[4]]
    Xlist=np.array([[C_fL[0],C_rL[0],C_wL[0],C_lL[0],C_sL[0]]])
    
    for x in xrange(1,i):

        C_f.append((1-d.p_5)*C_f[x-1]+d.p_3*(1-d.p_2)*ACM(C_f[x-1], x-1)[0])
        C_fL.append((1-d.p_5)*C_fL[x-1]+d.p_3*(1-d.p_2)*ACM(C_f[x-1], x-1)[1]*C_fL[x-1]) 
        C_rL.append((1-d.p_7)*C_rL[x-1]+(d.p_4*(1-d.p_3)*(1-d.p_2))*ACM(C_f[x-1], x-1)[1]*C_fL[x-1])
        C_wL.append((1-d.p_6)*C_wL[x-1]+((1-d.p_4)*(1-d.p_3)*(1-d.p_2))*ACM(C_f[x-1], x-1)[1]*C_fL[x-1])     
        C_lL.append((1-(d.p_1+d.p_8)*d.T[x-1])*C_lL[x-1]+(d.p_5)*C_fL[x-1]+(d.p_7)*C_rL[x-1])
        C_sL.append((1-d.p_9*d.T[x-1])*C_sL[x-1]+(d.p_6)*C_wL[x-1]+(d.p_1*d.T[x-1])*C_rL[x-1])       
        Xlist=np.append(Xlist, np.array([[C_fL[x],C_rL[x],C_wL[x],C_lL[x],C_sL[x]]]), axis=0)

    return Xlist

 
#Plot GPP and GPP_diff   
def GPP_plot(x0,i):
    Clist=nldalec(x0,i)[1]
    Cf=Clist[:,0]
    GPP=[None]*i
    for x in range(i):
        GPP[x]=ACM(Cf[x],x)
    xlist=np.arange(0,i,1)
    plt.plot(xlist,GPP)
    plt.show()    
    
	
def f(x,i):
    g=[x]
    for x in xrange(1,i):
        g.append((1-0.005)*g[x-1]+0.42*g[x-1]**2)

    return g    
	
def f_diff(x, dx, i):
    g=[x]
    gL=[dx]
	
    for x in xrange(1,i):
        g.append((1-0.005)*g[x-1]+0.42*g[x-1]**2)   
        gL.append((1-0.005)*gL[x-1]+0.42*2*g[x-1]*gL[x-1])
		
    return gL


