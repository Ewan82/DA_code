import numpy as np
f=open("dalec_drivers2.txt","r")
allLines=f.readlines()
lenrun=400
class dalecData( ): 

  def __init__(self):
    
    #'Driving data'
    self.data=np.matrix([[None]*9 for i in range(lenrun)])
    self.gpp=[None]*lenrun
    self.gpp_err=[None]*lenrun
    self.lf=[None]*lenrun
    self.lf_err=[None]*lenrun
    self.lw=[None]*lenrun
    self.lw_err=[None]*lenrun
    self.rt=[None]*lenrun
    self.rt_err=[None]*lenrun
    self.nee=[None]*lenrun
    self.nee_err=[None]*lenrun
    self.cf=[None]*lenrun
    self.cf_err=[None]*lenrun

    n=-1
    for x in xrange(0,lenrun):
        n=n+1
        allVars=allLines[x].split()
        for i in xrange(0,9):
            self.data[n,i]=float(allVars[i])
        for i in xrange(9,len(allVars)):
            if allVars[i]=='gpp':
                self.gpp[n]=float(allVars[i+1])
                self.gpp_err[n]=float(allVars[i+2])
            elif allVars[i]=='lf':
                self.lf[n]=float(allVars[i+1])
                self.lf_err[n]=float(allVars[i+2])
	    elif allVars[i]=='lw':
                self.lw[n]=float(allVars[i+1])
                self.lw_err[n]=float(allVars[i+2])
            elif allVars[i]=='rt':
                self.rt[n]=float(allVars[i+1])
                self.rt_err[n]=float(allVars[i+2])
            elif allVars[i]=='nee':
                self.nee[n]=float(allVars[i+1])
                self.nee_err[n]=float(allVars[i+2])
            elif allVars[i]=='cf':
                self.cf[n]=float(allVars[i+1])
                self.cf_err[n]=float(allVars[i+2])

    #'I.C. for carbon pools'
    self.C_f=np.array([58.0])
    self.C_r=np.array([102.0])
    self.C_w=np.array([770.0])
    self.C_l=np.array([40.0])
    self.C_s=np.array([9897.0])

    #'Background variances for carbon pools'
    self.sigB_cf=(self.C_f[0]*0.2)**2 #20%
    self.sigB_cw=(self.C_w[0]*0.2)**2 #20%
    self.sigB_cr=(self.C_r[0]*0.2)**2 #20%
    self.sigB_cl=(self.C_l[0]*0.2)**2 #20%
    self.sigB_cs=(self.C_s[0]*0.2)**2 #20% 

    #'Observartion variances for carbon pools and NEE' 
    self.sigO_cf=(self.C_f[0]*0.1)**2 #10%
    self.sigO_cw=(self.C_w[0]*0.1)**2 #10%
    self.sigO_cr=(self.C_r[0]*0.3)**2 #30%
    self.sigO_cl=(self.C_l[0]*0.3)**2 #30%
    self.sigO_cs=(self.C_s[0]*0.3)**2 #30% 


    #'Daily temperatures'
    self.T_mean=self.data[:,1]
    self.T_max=np.array(self.data[:,2])
    self.T_min=np.array(self.data[:,3])

    self.T_range=self.T_max-self.T_min
    self.T=[None]*len(self.T_mean)
    for i in range(len(self.T_mean)):
        self.T[i]=0.5*(np.exp(0.0693*float(self.T_mean[i]))) #Temp term
    
    #'Driving data'
    self.a_1=2.155
    self.a_2=0.0142
    self.a_3=217.9
    self.a_4=0.980
    self.a_5=0.155
    self.a_6=2.653
    self.a_7=4.309
    self.a_8=0.060
    self.a_9=1.062
    self.a_10=0.0006
    self.I=self.data[:,4] #incident radiation
    self.phi_d=self.data[:,5] #max. soil leaf water potential difference
    self.R_tot=self.data[:,7] #total plant-soil hydrolic resistance
    self.C_a=355.0 #atmospheric carbon
    self.N=2.7 #average foliar N
    self.p_1=0.00000441
    self.p_2=0.47
    self.p_3=0.31
    self.p_4=0.43
    self.p_5=0.0027
    self.p_6=0.00000206
    self.p_7=0.00248
    self.p_8=0.0228
    self.p_9=0.00000265
    self.D=self.data[:,0]#day of year
    self.bigdelta=0.908 #latitutde of forest site in radians


