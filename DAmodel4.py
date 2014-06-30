import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import Data2 as D
d=D.dalecData()

def ACM(Cf,i):
	
    q=d.a_3-d.a_4
    L=(Cf/111.)
    e_0=((d.a_7*L**2)/(L**2+d.a_9))
    g_c=(((np.abs(float(d.phi_d[i])))**(d.a_10))/(0.5*float(d.T_range[i])+d.a_6*float(d.R_tot[i])))
    p=(((d.a_1*d.N*L)/g_c)*np.exp(d.a_8*float(d.T_max[i])))
    C_i=(0.5*(d.C_a+q-p+np.sqrt((d.C_a+q-p)**2-4*(d.C_a*q-p*d.a_3))))
    delta=(-0.408*np.cos(((360*(float(d.D[i])+10)*np.pi)/(365*180))))
    s=(24*np.arccos((-np.tan(d.bigdelta)*np.tan(delta)))/np.pi)
	
    def GPPfunc(Cf):
		
        #L=(Cf/111.)
        #e_0=((d.a_7*L**2)/(L**2+d.a_9))
        #GPP=((e_0*float(d.I[i])*g_c*(d.C_a-C_i)*(d.a_2*s+d.a_5))/(e_0*float(d.I[i])+g_c*(d.C_a-C_i))) #0.2*float(d.I[i])*(1-np.exp(-0.5*Cf/111.))
        #GPP_diff=((2*float(d.I[i])*d.a_7*d.a_9*((111.0*g_c*(d.C_a-C_i))**2)*(d.a_2*s+d.a_5)*Cf)/(((g_c*(d.C_a-C_i)+d.a_7*float(d.I[i]))*Cf**2+d.a_9*(111.0**2)*g_c*(d.C_a-C_i))**2))  #(0.2*0.5*float(d.I[i])/111.)*np.exp(-0.5*Cf/111.)
        GPP= (d.a_7*(Cf**2)*float(d.I[i])*g_c*(d.C_a-C_i)*(d.a_2*s+d.a_5))/((d.a_7*float(d.I[i])+g_c*(d.C_a-C_i))*(Cf**2)+d.a_9*(111.**2)*g_c*(d.C_a-C_i))
        
        return GPP 

    def GPPdifffunc(Cf):
		
        GPP_diff=(24642.*float(d.I[i])*d.a_7*Cf*(g_c**2)*((d.C_a-C_i)**2)*(d.a_2*s+d.a_5)*d.a_9)/((Cf**2+12321.*d.a_9)*(d.C_a-C_i)*g_c+float(d.I[i])*d.a_7*Cf**2)**2 #(2*Cf*float(d.I[i])*d.a_7*d.a_9*(111.**2)*(g_c**2)*((d.C_a-C_i)**2)*(d.a_2*s+d.a_5))/((d.a_7*float(d.I[i])+g_c*(d.C_a-C_i))*(Cf**2)+d.a_9*(111.**2)*g_c*(d.C_a-C_i))**2
        
        return GPP_diff
		
    GPP = GPPfunc(Cf)
    GPP_diff=  GPPdifffunc(Cf)
    GPP_diff2 = misc.derivative(GPPfunc,Cf) 
    return GPP, GPP_diff
	

#Non-linear DALEC model
def nldalec(X,i):

    C_f=[X[0]]
    C_r=[X[1]]
    C_w=[X[2]]
    C_l=[X[3]]
    C_s=[X[4]]
    Xlist=np.array([[C_f[0],C_r[0],C_w[0],C_l[0],C_s[0]]])
    nee=[-(1-d.p_2)*ACM(C_f[0], 0)[0]+d.p_8*d.T[0]*C_l[0]+d.p_9*d.T[0]*C_s[0]]
    
    for x in xrange(1,i):

        C_f.append((1-d.p_5)*C_f[x-1]+d.p_3*(1-d.p_2)*ACM(C_f[x-1], x-1)[0])
        C_r.append((1-d.p_7)*C_r[x-1]+d.p_4*(1-d.p_3)*(1-d.p_2)*ACM(C_f[x-1], x-1)[0])
        C_w.append((1-d.p_6)*C_w[x-1]+(1-d.p_4)*(1-d.p_3)*(1-d.p_2)*ACM(C_f[x-1],x-1)[0])     
        C_l.append((1-(d.p_1+d.p_8)*d.T[x-1])*C_l[x-1]+d.p_5*C_f[x-1]+d.p_7*C_r[x-1])
        C_s.append((1-d.p_9*d.T[x-1])*C_s[x-1]+d.p_6*C_w[x-1]+d.p_1*d.T[x-1]*C_r[x-1])
        nee.append(-(1-d.p_2)*ACM(C_f[x], x)[0]+d.p_8*d.T[x]*C_l[x]+d.p_9*d.T[x]*C_s[x])
        Xlist=np.append(Xlist, np.array([[C_f[x],C_r[x],C_w[x],C_l[x],C_s[x]]]), axis=0)
	    
    return Xlist, nee


#Linear DALEC model !!TEST MODEL!!
def ldalec(X, i):   

    C_f=[X[0]]
    Mlist=[None]*(i-1)
    GPP_diff=[None]*i

    for x in xrange(0,i-1):
        GPP_diff[x]=ACM(C_f[x], x)[1]
        C_f.append((1-d.p_5)*C_f[x]+d.p_3*(1-d.p_2)*ACM(C_f[x], x)[0])
        Mlist[x]=(np.matrix([[(1-d.p_5)+d.p_3*(1-d.p_2)*ACM(C_f[x], x)[1],0,0,0,0],[(d.p_4*(1-d.p_3)*(1-d.p_2))*ACM(C_f[x], x)[1],(1-d.p_7),0,0,0],[(1-d.p_4)*(1-d.p_3)*(1-d.p_2)*ACM(C_f[x], x)[1],0,(1-d.p_6),0,0],[d.p_5,d.p_7,0,1-(d.p_1+d.p_8)*d.T[x],0],[0,0,d.p_6,d.p_1*d.T[x],1-d.p_9*d.T[x]]]))

    return Mlist, GPP_diff
    
    
#Linear DALEC model !!TEST MODEL!!
def ldalec2(X, dX, i):   

    C_f=[X[0]]
    C_fL=[dX[0]]
    C_rL=[dX[1]]
    C_wL=[dX[2]]
    C_lL=[dX[3]]
    C_sL=[dX[4]]
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
    
def GPP_plot(x0,i):
    Clist=nldalecx2(x0,i)[0]
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


