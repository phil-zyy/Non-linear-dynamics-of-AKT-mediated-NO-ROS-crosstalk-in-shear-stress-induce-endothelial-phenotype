from cc3d.core.PySteppables import *
import numpy as np
import random
import math
from scipy.integrate import solve_ivp
from pathlib import Path

t_stable = 1200;    # 稳态开始时间
t_step = 60;    # 时间间隔/s

ktgs = 4.0
# 初始化剪切力、角频率、正弦振动的振幅系数
omega = 2*np.pi #Hz
sigma = 0.4 #无量纲

mu1=0.2; rr=10.00; 
aktt=0.1; enost=0.04; pkct=aktt;
kcatp=0.026;

camtot=30; bufftot=120; pip2t=10.0;

phiinf=0.1; kleak = 1e-7; delta=24; lambdacap=15; Kcoup=0.002; SGC0=0.1; 

kpip2=0.021; kmpip2=0.7024; kmakt= 0.1155;
kcam0=7.5; kmcam=0.01; keakt=0.004; kmeakt=keakt/18; kpt5cam=3;
kon=100; koff=300;
qmax=17.6; kcce=8*1e-7; cs0=2828; cex=1500;
kres=5; krel=6.64; kout=24.7; vr=3.5; 
VpPKC=0.12/60; kcavdec=VpPKC*1/9;
Gt=0.332;
ka = 0.017; kd=0.15;
alp=2.781/0.332;

api3k=2.5;

rcgmp=1.26; Vdg=0.0695; kmdg=2; NOdec=382; RNO=300.0;

eta=0.0030000;
k2=0.2; k3=0.15; k5=0.32; k2p=0.0022;
theta = 0.0045;
N = 2;
xi=0.15/20;
kmpkc = kmakt;
kdcam = 1;
kca4cam = kon;
a0 = 1200.1667*1e-6;
a1 = 37.34*1e-3;
g0 = 4.8006*1e-6;
g1 = 35.3*1e-3;
b1 = 15.15*1e-3;

NOD = 0.05
UNOMax = 0.11592 
NOK=0.04

TNFD = 3.6e-6
UTNFMax = 3.0e-5
TNFK=1.68e-5

UILMax = 3.0e-5
ILK = 1.92e-4
K = 1 #可以变换参数，从1-2看治疗效果，补充IL-10

decvol = 0.0167 # decreasing rate of tumor cells in pixels/MCS 
maxdiv = 6

class Endothelial_Layer_Initializer_Steppable(SteppableBasePy):
    
    def start(self):
        # 获取化学场对象
        ShearStress_field = self.field.ShearStressField  # Low pulsation flow shear stress    
        # 定义两边的最大浓度和中间的最小浓度
        max_ShearStress = 15.0  # 两边的最大浓度
        min_ShearStress = 5.0  # 中间的最低浓度
        width = self.dim.x       # 模拟区域的宽度
        center = width // 2      # 中心位置   
        # 计算从中间到两边的对称递增
        for x in range(width):
            # 计算 x 到中心的距离，并将其标准化到 [0, 1]
            distance_to_center = abs(x - center) / float(center)          
            # 调整浓度：中心浓度为 min_concentration，两边逐渐递增到 max_concentration
            concentration = min_ShearStress + (max_ShearStress - min_ShearStress) * distance_to_center           
            for y in range(0, 100):
                ShearStress_field[x, y, 0] = concentration  # 设置每个位置的浓度
        for cell in self.cell_list_by_type(self.QEC): 
            cell.targetVolume = 25
            cell.lambdaVolume = 500.0  # 调节体积约束强度的参数
            cell.targetSurface = 25 
            cell.lambdaSurface = 500.0   # 表面积大一点以维持形状  
            cell.dict["Counter"] = 0
    
    def step(self, mcs):
       
        for cell in self.cell_list_by_type(self.AEC, self.DEC, self.DEAD):
               
            if cell.type == self.AEC:      
                cell.lambdaVecX = 160*np.random.uniform(-1, 1)
                # force component along Y axis
                cell.lambdaVecY = 160*np.random.uniform(-1, 1)
                if cell.yCOM > 40:                    
                    cell.dict["apoptosis"] +=1
                elif cell.yCOM <= 40:
                    cell.lambdaVecY = 160*np.random.uniform(0, 1)
                    cell.targetVolume += 0.01389
                    cell.lambdaVolume = ktgs*sqrt(cell.targetVolume)
                    
            if cell.type == self.DEC: 
                cell.targetVolume -= min(decvol,cell.targetVolume)   
                cell.targetSurface=ktgs*sqrt(cell.targetVolume)                
                
            if cell.type == self.DEAD:
                cell.targetVolume -= min(decvol,cell.targetVolume)   
                cell.targetSurface=ktgs*sqrt(cell.targetVolume)    
                
            if cell.targetVolume<=0:
                self.delete_cell(cell) 
                
class ODESteppable(SteppableBasePy):
     
    def start(self):        

        self.time = 0; # 当前时间
        
        # 初始条件设置，例如细胞内部物质浓度的初值
        self.NO_Field = self.field.NO       
        for cell in self.cell_list_by_type(self.QEC):
               cell.dict['G'] = 0
               cell.dict['IP3'] = 0
               cell.dict['PIP2'] = 10
               cell.dict['PIP3'] = 0.0042
               cell.dict['CAc'] = 0.1
               cell.dict['CAs'] = 2828
               cell.dict['CAb'] = 0
               cell.dict['AKTa'] = 0.0031
               cell.dict['PKCa'] = 0.0031
               cell.dict['Ca4CaM'] = 0
               cell.dict['eNOS_CaM'] = 0
               cell.dict['eNOS_CaMa'] = 0
               cell.dict['eNOScav0'] = 0
               cell.dict['NO'] = 0
               cell.dict['cGMP'] = 0
               cell.dict['eNOScav'] = 0.04 
               cell.dict['ShearStress'] = self.field.ShearStressField[cell.xCOM, cell.yCOM, 0.0]
               
    def F(self, tau, omega, t): #  pulsatileWSS，可以换振荡和脉动流，作对比。同一个剪应力下，不用的流。先搞同一流下，不同力的大小

        return tau * np.cos(omega*t)  # 
        
    def rstar(self, tau, omega, t, lambdacap):
        
        return 0.5 * (np.tanh(3.14 * self.F(tau, omega, t) / lambdacap) + 1) 
    
    def W(self, tau, omega, t):
        return 1.4*(self.F(tau, omega, t)/10+np.sqrt(16*2.86**2+self.F(tau, omega, t)**2/100)-4*2.86)**2/(self.F(tau, omega, t)/10+np.sqrt(16*2.86**2+self.F(tau, omega, t)**2/100))   
        
    def ode_system(self, t, y, tau, omega, lambdacap):
        
        if not isinstance(y, (list, np.ndarray)):
            raise TypeError("y must be a list or numpy array")
        
        # 解包 y
        G, IP3, PIP2, PIP3, CAc, CAs, CAb, AKTa, PKCa, Ca4CaM, eNOS_CaM, eNOS_CaMa, eNOScav0, NO, cGMP, eNOScav = y
            
        Beta    = 2.7;
        Psi     = 1.00 - xi*cGMP; 
        Gd = Gt-G
        rf = alp*Kcoup*phiinf/(phiinf+kcatp)*G;
        kp_pip2 = kpip2/(1 + api3k)* (1+api3k*np.exp(-eta*t)*np.tanh(np.pi*self.F(tau, omega, t)/delta)) + k2p;
            
        qrel    = krel*(IP3/(k2+IP3))**3*CAs;  
        qres    = kres*(CAc/(k3+CAc))**2;
        qout    = kout*(CAc/(k5+CAc));
        qmsic   = qmax/(1+N*np.exp(-self.W(tau, omega, t)));
        qcce    = kcce*Psi*(cs0 - CAs)*(cex - CAc);
        qin     = qmsic + qcce;
        qon     = kon*CAc*(bufftot-CAb);
        qoff    = koff*CAb;
            
        PIP3p   = PIP3/pip2t*100;
        if  self.mcs <= 4320:
            kpakt   = 0.1*kmakt*(PIP3p-0.31)/(3.1-0.31) #kpakt扩大1.8倍看一下
        #elif 1440 < self.mcs <= 2880: 
            #kpakt   = 0.1*kmakt*(PIP3p-0.31)/(3.1-0.31)  #修复变量
        else :  
             kpakt   = 0.1*kmakt*(PIP3p-0.31)/(3.1-0.31) 
        #kpakt   = 0.4*kmakt*(PIP3p-0.31)/(3.1-0.31);       
        kppkc   = 0.1*kmpkc*(PIP3p-0.31)/(3.1-0.31);
        PKC     = pkct - PKCa;
        AKT     = aktt - AKTa;
        
        dGdt = ka*Gd*self.rstar(tau, omega, t, lambdacap) - kd*G
            
        dIP3dt = rf*PIP2 - mu1*IP3;
            
        dPIP2dt = -(rf+rr)*PIP2 - rr*IP3 + rr*pip2t-(PIP2*(kp_pip2) - kmpip2*PIP3)
            
        dPIP3dt = PIP2*(kp_pip2) - kmpip2*PIP3;
            
        dCAcdt = qrel - qres - qout + qin + kleak*(CAs)**2 - qon + qoff;
            
        dCAsdt = -vr*(qrel - qres +kleak*(CAs)**2);
            
        dCAbdt = qon - qoff;
            
        dAKTadt = kpakt*AKT - kmakt*AKTa; 
            
        dPKCadt = kppkc*PKC - kmpkc*PKCa;
            
        dCa4CaMdt = kca4cam*(camtot*theta*CAc**Beta/(kdcam+CAc**Beta) - Ca4CaM);
            
        deNOS_CaMdt = kcam0*Ca4CaM/(kpt5cam+Ca4CaM)*eNOScav - kmcam*eNOS_CaM - (keakt*eNOS_CaM*AKTa/aktt - kmeakt*eNOS_CaMa); 
            
        deNOS_CaMadt = keakt*eNOS_CaM*AKTa/aktt - kmeakt*eNOS_CaMa; 
            
        deNOScav0dt = VpPKC*PKCa/pkct*(eNOScav) - kcavdec*eNOScav0;
            
        dNOdt = RNO*(eNOS_CaM+9*eNOS_CaMa) - NOdec*NO - 0.022*SGC0*(NO**2 + NO*b1)/(a0+a1*NO+NO**2);
            
        dcGMPdt = rcgmp*(g0+g1*NO+(NO)**2)/(a0+a1*NO+(NO)**2) - (cGMP)**2*Vdg/(kmdg+cGMP);
            
        deNOScavdt = - (VpPKC*PKCa/pkct*(eNOScav) - kcavdec*eNOScav0) - (kcam0*Ca4CaM/(kpt5cam+Ca4CaM)*eNOScav - kmcam*eNOS_CaM);  
        
        return [dGdt, dIP3dt, dPIP2dt, dPIP3dt, 
                dCAcdt, dCAsdt, dCAbdt, dAKTadt, dPKCadt, 
                dCa4CaMdt, deNOS_CaMdt, deNOS_CaMadt, deNOScav0dt, 
                dNOdt, dcGMPdt, deNOScavdt]  
                
    def solve_analytic(self, _t0, _T, _y0, _tau):
    
        _t = _T-_t0;
    
        G_0, IP3_0, PIP2_0, PIP3_0, CAc_0, CAs_0, CAb_0, AKTa_0, PKCa_0, Ca4CaM_0, eNOS_CaM_0, eNOS_CaMa_0, eNOScav0_0, NO_0, cGMP_0, eNOScav_0 = _y0;
    
    
        # *****************************************************
        # Ga
        # *****************************************************
    
        _r_star_bar = 0.5;
        _ga_bar = Gt / (1+kd/(ka*_r_star_bar));
        _ga_lambda = -(ka*_r_star_bar+kd);
        _ga = (G_0-_ga_bar)*np.exp(_ga_lambda*_t)+_ga_bar;
    
    
        # *****************************************************
        # PIP2 - PIP3 - IP3
        # *****************************************************
    
        # PIP2:
        _k1 = alp*Kcoup*phiinf/(phiinf+kcatp);
        _k2 = rr;
        _kd1 = mu1;
    
    #kpip2/(1 + api3k)* (1+api3k*np.exp(-eta*t)*np.tanh(np.pi*F(t)/delta)) + k2p;
        _exp_eta_bar = (np.exp(-eta*_t0)-np.exp(-eta*_T)) / (eta*(_T-_t0));
        _r_star_bar_pip2 = 0;
        _kf9f10_stable = k2p + kpip2/(1 + api3k);
        _ep_kf9f10 = kpip2/(1 + api3k) * api3k * _r_star_bar_pip2;
        _kf9_kf10_bar = _kf9f10_stable + _ep_kf9f10 * _exp_eta_bar;
        _lambda_pip2 = -(_k1*_ga_bar + _k2 + _kf9_kf10_bar);
    
        _pip2_stable = _k2*pip2t / ((_k1*_k2*_ga_bar)/_kd1 - _kf9_kf10_bar - _lambda_pip2);
        #_pip2_stable = _k2*pip2t / ((_k1*_k2*_ga_bar)/_kd1 - _kf9f10_stable - _lambda_pip2);
        _h_pip2 = PIP2_0 - _pip2_stable;
        _pip2 = _pip2_stable + _h_pip2 * np.exp(_lambda_pip2*_t);
    
        # PIP3:
        _lambda_pip3 = -kmpip2;
        _pip3_stable = _kf9f10_stable/(-_lambda_pip3) * _pip2_stable;
        _h0_pip3 = -(_ep_kf9f10 * _pip2_stable) / (eta + _lambda_pip3);
        _h1_pip3 = (_kf9f10_stable * _h_pip2) / (_lambda_pip2 - _lambda_pip3);
        _h2_pip3 = _h_pip2 / (_lambda_pip2 - _lambda_pip3 - eta);
    
        #_h3_pip3 = (PIP3_0 - _pip3_stable -_h0_pip3*np.exp(-eta*_t0) - _h1_pip3*np.exp(_lambda_pip2 * _t0) - _h2_pip3*np.exp((_lambda_pip2 - eta) * _t0)) / np.exp(_lambda_pip3*_t0);
        #_pip3 = _pip3_stable + _h0_pip3 * np.exp(-eta*(_t+_t0)) + _h1_pip3 * np.exp(_lambda_pip2 * (_t+_t0)) + _h2_pip3 * np.exp((_lambda_pip2 - eta) * (_t+_t0)) + _h3_pip3 * np.exp(_lambda_pip3*(_t+_t0));
        _h3_pip3 = PIP3_0 - _pip3_stable -_h0_pip3*np.exp(-eta*_t0);
        _pip3 = _pip3_stable + _h0_pip3 * np.exp(-eta*(_t+_t0)) + _h3_pip3 * np.exp(_lambda_pip3*_t);
    
        # IP3:
        _lambda_ip3 = -_kd1;
        _ip3_stable = _k1*_ga_bar/(-_lambda_ip3) * _pip2_stable;
        _h1_ip3 = (_k1 * _ga_bar * _h_pip2) / (_lambda_pip2 - _lambda_ip3);
        _h2_ip3 = IP3_0 - _ip3_stable - _h1_ip3;
        _ip3 = _ip3_stable + _h1_ip3 * np.exp(_lambda_pip2*_t) + _h2_ip3 * np.exp(_lambda_ip3*_t);
    
    
        # *****************************************************
        # PKC / AKT
        # *****************************************************
    
        _pip3_min = 0.0031*pip2t;
        _pip3_max = 0.031*pip2t;
        _kr13 = kmpkc;
        _kr14 = kmakt;
        _kf13 = 0.1*kmpkc;
        if  self.mcs <= 4320:
             _kf14 = 0.1*kmakt #kpakt扩大1.8倍看一下
       # elif  1440< self.mcs <= 2880:  #修复变量
            #_kf14 = 0.1*kmakt 
        else  :
           _kf14 = 0.1 * kmakt 
        #_kf14 = 0.1*kmakt; #kpakt扩大1.8倍
    
        _kp_pkc_bar = _kf13 * (_pip3_stable - _pip3_min) / (_pip3_max - _pip3_min);
        _kp_akt_bar = _kf14 * (_pip3_stable - _pip3_min) / (_pip3_max - _pip3_min);
        _H0_pip3_pkc = (_kf13 * _h0_pip3) / (_pip3_max - _pip3_min);
        _H0_pip3_akt = (_kf14 * _h0_pip3) / (_pip3_max - _pip3_min);
        _H3_pip3_pkc = (_kf13 * _h3_pip3) / (_pip3_max - _pip3_min);
        _H3_pip3_akt = (_kf14 * _h3_pip3) / (_pip3_max - _pip3_min);
        _kp_pkc_tilde = _kp_pkc_bar + _kr13 + _H0_pip3_pkc * _exp_eta_bar;
        _kp_akt_tilde = _kp_akt_bar + _kr14 + _H0_pip3_akt * _exp_eta_bar;
        _pkc_stable = (_kp_pkc_bar * pkct) / _kp_pkc_tilde;
        _akt_stable = (_kp_akt_bar * aktt) / _kp_akt_tilde;
    
        _h0_pkc = (_H0_pip3_pkc * pkct) / (_kp_pkc_tilde - eta);
        _h0_akt = (_H0_pip3_pkc * aktt) / (_kp_akt_tilde - eta);
        _h3_pkc = (_H3_pip3_pkc * pkct) / (_kp_pkc_tilde + _lambda_pip3);
        _h3_akt = (_H3_pip3_pkc * aktt) / (_kp_akt_tilde + _lambda_pip3);
        _h4_pkc = PKCa_0 - _pkc_stable - _h0_pkc*np.exp(-eta*_t0) - _h3_pkc;
        _h4_akt = AKTa_0 - _akt_stable - _h0_akt*np.exp(-eta*_t0) - _h3_akt;
    
        _pkc = _pkc_stable + _h0_pkc * np.exp(-eta*(_t+_t0)) + _h3_pkc * np.exp(_lambda_pip3*_t) + _h4_pkc * np.exp(-_kp_pkc_tilde*_t);
        _akt = _akt_stable + _h0_akt * np.exp(-eta*(_t+_t0)) + _h3_akt * np.exp(_lambda_pip3*_t) + _h4_akt * np.exp(-_kp_akt_tilde*_t);
    
    
        # *****************************************************
        # CA Series
        # *****************************************************
    
        _k_ip3 = krel*(IP3_0/(k2+IP3_0))**3;
    
        _f1x1_ca = -kon * (bufftot - CAb_0) - kcce * (1 - xi*cGMP_0) * (cs0 - CAs_0) - 2 * kres * CAc_0/(k3 + CAc_0) * (1/(k3 + CAc_0) - CAc_0/(k3 + CAc_0)**2) - kout * (1/(k5+CAc_0) - CAc_0/(k5 + CAc_0)**2);
        _f1x2_ca = _k_ip3 + 2 * kleak * CAs_0 - kcce * (1 - xi*cGMP_0) * (cex - CAc_0);
        _f1x3_ca = koff + kon * CAc_0;
        _f2x1_ca = 2 * vr * kres * CAc_0/(k3 + CAc_0) * (1/(k3 + CAc_0) - CAc_0/(k3 + CAc_0)**2);
        _f2x2_ca = -vr * (_k_ip3 + 2 * kleak * CAs_0);
        _f3x1_ca = kon * (bufftot - CAb_0);
        _f3x3_ca = -(kon * CAc_0 + koff);
    
        _A_ca = np.array([
            [_f1x1_ca, _f1x2_ca, _f1x3_ca],
            [_f2x1_ca, _f2x2_ca, 0],
            [_f3x1_ca, 0, _f3x3_ca],
        ]);
    
        _eigen_ca, _V_ca = np.linalg.eig(_A_ca)
        _diag_ca = np.diag(_eigen_ca)
    
        _qrel0    = _k_ip3*CAs_0;
        _qres0    = kres*(CAc_0/(k3+CAc_0))**2;
        _qout0    = kout*(CAc_0/(k5+CAc_0));
        _qmsic0   = 13.545/(1+1.307*np.exp(-5.957e-4*_tau**2.072));
        _qcce0    = kcce*(1 - xi*cGMP_0)*(cs0 - CAs_0)*(cex - CAc_0);
        _qin0     = _qmsic0 + _qcce0;
        _qon0     = kon*CAc_0*(bufftot-CAb_0);
        _qoff0    = koff*CAb_0;
        _dCAcdt = _qrel0 - _qres0 - _qout0 + _qin0 + kleak*CAs_0**2 - _qon0 + _qoff0;
        _dCAsdt = -vr*(_qrel0 - _qres0 +kleak*CAs_0**2);
        _dCAbdt = _qon0 - _qoff0;
    
        _B_ca = np.array([_dCAcdt, _dCAsdt, _dCAbdt]) - _A_ca @ np.array([CAc_0, CAs_0, CAb_0]);
        _ca_stable = -np.linalg.inv(_A_ca) @ _B_ca;
        _x0_ca_v = np.array([CAc_0, CAs_0, CAb_0]) - _ca_stable;
        _z0_ca_v = np.linalg.inv(_V_ca) @ _x0_ca_v;
    
        _cac = _V_ca[0,0]*_z0_ca_v[0]*np.exp(_eigen_ca[0]*_t) + _V_ca[0,1]*_z0_ca_v[1]*np.exp(_eigen_ca[1]*_t) + _V_ca[0,2]*_z0_ca_v[2]*np.exp(_eigen_ca[2]*_t) + _ca_stable[0];
        _cas = _V_ca[1,0]*_z0_ca_v[0]*np.exp(_eigen_ca[0]*_t) + _V_ca[1,1]*_z0_ca_v[1]*np.exp(_eigen_ca[1]*_t) + _V_ca[1,2]*_z0_ca_v[2]*np.exp(_eigen_ca[2]*_t) + _ca_stable[1];
        _cab = _V_ca[2,0]*_z0_ca_v[0]*np.exp(_eigen_ca[0]*_t) + _V_ca[2,1]*_z0_ca_v[1]*np.exp(_eigen_ca[1]*_t) + _V_ca[2,2]*_z0_ca_v[2]*np.exp(_eigen_ca[2]*_t) + _ca_stable[2];
        #_cac = CAc_0;
        #_cas = CAs_0;
        #_cab = CAb_0;
    
        ## CAs 第二种算法
        #_B_cac = kleak*kres*((vr*CAc_0/(k3+CAc_0))**2);
        #_delta_cac = np.sqrt((vr*_k_ip3)**2 + 4*_B_cac);
        #_r1_cac = (-vr*_k_ip3-_delta_cac) / 2;
        #_r2_cac = (-vr*_k_ip3+_delta_cac) / 2;
        #_h_cac = _delta_cac / (_r2_cac - vr*kleak*CAs_0) - 1;
        #_cas = _r2_cac / (vr*kleak) - _delta_cac / (vr*kleak*(1+_h_cac*np.exp(_delta_cac*_t)));
    
        # Ca4-CaM
        _lambda_ca4cam = -kca4cam;
        _bata = 2.7;
        _b_ca4cam = kca4cam * camtot*theta*CAc_0**_bata/(kdcam+CAc_0**_bata);
        _h_ca4cam = Ca4CaM_0 + _b_ca4cam/_lambda_ca4cam;
        _ca4cam = -_b_ca4cam/_lambda_ca4cam + _h_ca4cam*np.exp(_lambda_ca4cam*_t);
    
    
        # *****************************************************
        # eNOS Series
        # *****************************************************
    
        _kr17 = kcavdec;
        _kf17 = VpPKC;
        _kf17_tilde = _kf17*_pkc/pkct;
    
        _kr16 = kmeakt;
        _kf16 = keakt;
        _kf16_tilde = _kf16*_akt/aktt;
    
        _kr15 = kmcam;
        _kf15 = kcam0;
        _kf15_tilde = _kf15*_ca4cam/(kpt5cam+_ca4cam);
    
        _A_enos = np.array([
            [-_kr17, _kf17_tilde, 0, 0],
            [_kr17, -(_kf17_tilde+_kf15_tilde), _kr15, 0],
            [0, _kf15_tilde, -(_kr15+_kf16_tilde), _kr16],
            [0, 0, _kf16_tilde, -_kr16],
        ]);
    
        _eigen_enos, _V_enos = np.linalg.eig(_A_enos)
        _diag_enos = np.diag(_eigen_enos)
    
        _x0_enos_v = np.array([eNOScav0_0, eNOScav_0, eNOS_CaM_0, eNOS_CaMa_0]);
        _z0_enos_v = np.linalg.inv(_V_enos) @ _x0_enos_v;
    
        _enos_cav0 = _V_enos[0,0]*_z0_enos_v[0]*np.exp(_eigen_enos[0]*_t) + _V_enos[0,1]*_z0_enos_v[1]*np.exp(_eigen_enos[1]*_t) + _V_enos[0,2]*_z0_enos_v[2]*np.exp(_eigen_enos[2]*_t) + _V_enos[0,3]*_z0_enos_v[3]*np.exp(_eigen_enos[3]*_t);
        _enos_cav1 = _V_enos[1,0]*_z0_enos_v[0]*np.exp(_eigen_enos[0]*_t) + _V_enos[1,1]*_z0_enos_v[1]*np.exp(_eigen_enos[1]*_t) + _V_enos[1,2]*_z0_enos_v[2]*np.exp(_eigen_enos[2]*_t) + _V_enos[1,3]*_z0_enos_v[3]*np.exp(_eigen_enos[3]*_t);
        _enos_cam0 = _V_enos[2,0]*_z0_enos_v[0]*np.exp(_eigen_enos[0]*_t) + _V_enos[2,1]*_z0_enos_v[1]*np.exp(_eigen_enos[1]*_t) + _V_enos[2,2]*_z0_enos_v[2]*np.exp(_eigen_enos[2]*_t) + _V_enos[2,3]*_z0_enos_v[3]*np.exp(_eigen_enos[3]*_t);
        _enos_cama = _V_enos[3,0]*_z0_enos_v[0]*np.exp(_eigen_enos[0]*_t) + _V_enos[3,1]*_z0_enos_v[1]*np.exp(_eigen_enos[1]*_t) + _V_enos[3,2]*_z0_enos_v[2]*np.exp(_eigen_enos[2]*_t) + _V_enos[3,3]*_z0_enos_v[3]*np.exp(_eigen_enos[3]*_t);
    
    
        # *****************************************************
        # NO & cGMP
        # *****************************************************
    
        _h1_enos = (RNO * _z0_enos_v[0] * (_V_enos[2,0] + 9 * _V_enos[3,0])) / (NOdec + _eigen_enos[0]);
        _h2_enos = (RNO * _z0_enos_v[1] * (_V_enos[2,1] + 9 * _V_enos[3,1])) / (NOdec + _eigen_enos[1]);
        _h3_enos = (RNO * _z0_enos_v[2] * (_V_enos[2,2] + 9 * _V_enos[3,2])) / (NOdec + _eigen_enos[2]);
        _h4_enos = (RNO * _z0_enos_v[3] * (_V_enos[2,3] + 9 * _V_enos[3,3])) / (NOdec + _eigen_enos[3]);
        _lambda_no = -NOdec - 0.022*SGC0 * ((2*NO_0 + b1)/(a0+a1*NO_0+NO_0**2) - (NO_0**2 + NO_0*b1)*(a1+2*NO_0)/(a0+a1*NO_0+NO_0**2)**2);
    
        _h_no = NO_0 - _h1_enos - _h2_enos - _h3_enos - _h4_enos;
        _no = _h_no * np.exp(_lambda_no * _t) + _h1_enos * np.exp(_eigen_enos[0]*_t) + _h2_enos * np.exp(_eigen_enos[1]*_t) + _h3_enos * np.exp(_eigen_enos[2]*_t) + _h4_enos * np.exp(_eigen_enos[3]*_t);
    
        _B_cgmp = rcgmp * (g0 + g1*NO_0 + NO_0**2) / (a0 + a1*NO_0 + NO_0**2);
    
        _c1_cgmp = -(kmdg * cGMP_0) / (kmdg + cGMP_0);
        _c2_cgmp = -(_B_cgmp*kmdg + _B_cgmp*cGMP_0 - (cGMP_0**2 * Vdg)) / ((kmdg + cGMP_0) * Vdg);
        _cgmp = (_B_cgmp - Vdg*_c1_cgmp) / Vdg + _c2_cgmp * np.exp(-Vdg * _t);
    
        #print(f'Used {datetime.now() - _start_time} for {len(_t)} points.');
    
        return [_ga, _ip3, _pip2, _pip3, _cac, _cas, _cab, _akt, _pkc, _ca4cam, _enos_cam0, _enos_cama, _enos_cav0, _no, _cgmp, _enos_cav1];

    def step(self, mcs):
        
        total_ANO = 0.0
        total_AROS = 0.0
        total_QNO = 0.0  
        
        NO_secretor = self.get_field_secretor("NO")
        # 时间点设置（例如模拟的步长）
        t_span = (self.time, self.time + t_step)
        t_eval = np.linspace(self.time, self.time + t_step, 121)
        #t = np.linspace(0, t_step, 601)  # 时间点，0到1之间np.linspace(0, t_step, 120)  改ODE中的参数单位         
        for cell in self.cell_list_by_type(self.QEC, self.AEC):
            
            tau = cell.dict['ShearStress']
            #print(tau)                
            # 获取当前细胞的初始浓度
            if cell.type == self.QEC:  
                y0 = [cell.dict['G'], cell.dict['IP3'], cell.dict['PIP2'], cell.dict['PIP3'],
                      cell.dict['CAc'], cell.dict['CAs'], cell.dict['CAb'], cell.dict['AKTa'], cell.dict['PKCa'],
                      cell.dict['Ca4CaM'], cell.dict['eNOS_CaM'], cell.dict['eNOS_CaMa'], cell.dict['eNOScav0'],
                      cell.dict['NO'], cell.dict['cGMP'], cell.dict['eNOScav']]        
                # 求解ODE系统
                if self.time <= t_stable:
                    sol_num = solve_ivp(self.ode_system, t_span, y0,  method='BDF', t_eval=t_eval, args = (tau,omega,lambdacap)) # t_stable之前求数值解
                else:
                    sol_anl = self.solve_analytic(self.time, self.time+t_step, y0, tau); # t_stable之后求解析解
                 # 更新细胞的属性，将ODE求解的最终浓度值应用到细胞上
                keys = ['G', 'IP3', 'PIP2', 'PIP3', 'CAc', 'CAs', 'CAb', 'AKTa', 'PKCa', 'Ca4CaM', 'eNOS_CaM', 'eNOS_CaMa', 'eNOScav0', 'NO', 'cGMP', 'eNOScav']
                # 更新细胞的属性，将ODE求解的最终浓度值应用到细胞上
                for i, key in enumerate(keys):
                    if self.time <= t_stable:
                        cell.dict[key] = sol_num.y[i][-1]
                    else:
                        cell.dict[key] = sol_anl[i]
                NO_secretor.secreteInsideCell(cell, cell.dict['NO']) 
                
                total_QNO += cell.dict['NO']
                
                if  mcs % 360 == 0:
                    output_dir = self.output_dir
                    if output_dir is not None:
                        output_path = Path(output_dir).joinpath('QECNO' + '.csv')
                        with open(output_path, 'a+') as fout:
                            fout.write('{} {} {}\n'.format(mcs, cell.id, cell.dict['NO']))       
                
            elif cell.type == self.AEC:  #细胞类型转换后需要重新给字典赋值
                # 检查是否已经初始化NewNO
                y0 = [cell.dict['G'], cell.dict['IP3'], cell.dict['PIP2'], cell.dict['PIP3'],
                      cell.dict['CAc'], cell.dict['CAs'], cell.dict['CAb'], cell.dict['AKTa'], cell.dict['PKCa'],
                      cell.dict['Ca4CaM'], cell.dict['eNOS_CaM'], cell.dict['eNOS_CaMa'], cell.dict['eNOScav0'],
                      cell.dict['NO'], cell.dict['cGMP'], cell.dict['eNOScav']] 
                # 求解ODE系统
                if self.time <= t_stable:
                    sol_num = solve_ivp(self.ode_system, t_span, y0,  method='BDF', t_eval=t_eval, args = (tau,omega,lambdacap)) # t_stable之前求数值解
                else:
                    sol_anl = self.solve_analytic(self.time, self.time+t_step, y0, tau); # t_stable之后求解析解
                 # 更新细胞的属性，将ODE求解的最终浓度值应用到细胞上
                keys = ['G', 'IP3', 'PIP2', 'PIP3', 'CAc', 'CAs', 'CAb', 'AKTa', 'PKCa', 'Ca4CaM', 'eNOS_CaM', 'eNOS_CaMa', 'eNOScav0', 'NO', 'cGMP', 'eNOScav']
                # 更新细胞的属性，将ODE求解的最终浓度值应用到细胞上
                for i, key in enumerate(keys):
                    if self.time <= t_stable:
                        cell.dict[key] = sol_num.y[i][-1]
                    else:
                        cell.dict[key] = sol_anl[i]
                #NO_secretor.secreteInsideCell(cell, cell.dict['NO']) 
                if not cell.dict.get('initialized', False):
                    cell.dict['NewNO'] = cell.dict['NO']  # 第一次进入时初始化NewNO
                    cell.dict['initialized'] = True      # 设置标志位为True
                # 计算阈值（NO值的60%）
                threshold = 0.6 * cell.dict['NO']
                # 根据NewNO的值进行更新和分泌
                if cell.dict['NewNO'] > threshold:                                   
                    cell.dict['NewNO'] = threshold + (cell.dict['NewNO'] - threshold) * np.exp(- 1e-5 * cell.dict['current_mcs']) # 使用独立计时器
                    cell.dict['current_mcs'] += 1
                    #cell.dict['NewNO'] =  (cell.dict['NewNO'] - threshold) * np.exp(- 10 * mcs)             
                    NO_secretor.secreteInsideCell(cell, cell.dict['NewNO'])
                else:
                    # 分泌NO值的60%
                    cell.dict['NewNO'] = threshold
                    NO_secretor.secreteInsideCell(cell, cell.dict['NewNO'])
                    
                total_ANO += cell.dict['NewNO']
                total_AROS += cell.dict["ROS"]  
                
                if  mcs % 360 == 0:    
                    output_dir = self.output_dir
                    if output_dir is not None:
                        output_path = Path(output_dir).joinpath('AECNO' + '.csv')
                        with open(output_path, 'a+') as fout:
                            fout.write('{} {} {} \n'.format(mcs, cell.id, cell.dict['NewNO']))                        
            #print(cell.dict['NO'])                            
            total_NO =  total_ANO +  total_QNO
            # 防止除零
            if total_NO > 0:
                a = total_ANO / total_NO
                b = total_QNO / total_NO
            else:
                a = 0
                b = 0
        
        output_dir = self.output_dir
        if output_dir is not None:
            output_path = Path(output_dir).joinpath('totalNOROS' + '.csv')
            with open(output_path, 'a+') as fout:
                fout.write('{} {} {} {} {} {} {}\n'.format(mcs, total_ANO, total_QNO, total_NO, a, b, total_AROS)) 
                
        self.time = self.time + t_step; # 迭代初始时间

class ROSfield(SteppableBasePy): 
       
    def step(self,mcs):
        NO_field = self.field.NO
        ROS_field = self.field.ROS
        ROS_secretor = self.get_field_secretor("ROS")
        for cell in self.cell_list_by_type(self.AEC, self.M1):
            if cell.type == self.AEC:
                ROS_secretor.secreteInsideCell(cell, 0.13*(1+cell.dict["Damage"]/(cell.dict["Damage"]+400)))
                cell.dict["ROS"] = 0.13*(1+cell.dict["Damage"]/(cell.dict["Damage"]+400))
                if  mcs % 360 == 0:    
                    output_dir = self.output_dir
                    if output_dir is not None:
                        output_path = Path(output_dir).joinpath('AECROS' + '.csv')
                        with open(output_path, 'a+') as fout:
                            fout.write('{} {} {} \n'.format(mcs, cell.id, cell.dict["ROS"])) 
            elif cell.type == self.M1:
                ROS_secretor.secreteInsideCell(cell, 0.13)                
            for pixel in self.getCellPixelList(cell):
                x = pixel.pixel.x
                y = pixel.pixel.y
                z = pixel.pixel.z    
                NO = NO_field[x,y,z]
                ROS = ROS_field[x,y,z]        
                reaction = 1 * NO * ROS/(ROS+0.6)       
                NO_field[x,y,z]  = max(0.0, NO - reaction)
                ROS_field[x,y,z] = max(0.0, ROS - reaction)
         
                        
class DamageProtectionCalculatorSteppable(SteppableBasePy): 
    
    def start(self):
            
        for cell in self.cell_list_by_type(self.QEC, self.AEC):
            cell.dict["Damage"] = 0
            cell.dict["Health"] = 0

    def MM(self,x,m,k,a):
        return (m*x**a/(x**a + k**a))
    
    def step(self,mcs):
        #200 mcs之后再开始启动累加损伤程序  
        if mcs > 200: 
            NO_field = self.field.NO #new
            
            for cell in self.cell_list_by_type(self.QEC, self.AEC): 
                NO_conc = NO_field[cell.xCOM, cell.yCOM, 0.0]   
                if cell.type == self.QEC:
                    #NO_conc = cell.dict["NO"]
                    # Damage pathway
                    if NO_conc < NOD: 
                        cell.dict["Damage"] += abs(self.MM(NO_conc, UNOMax, NOK, 4) - self.MM(NOD, UNOMax, NOK, 4))                                        
                    # Healthy pathway
                    if NO_conc >= NOD:                           
                        cell.dict["Health"] += abs(self.MM(NO_conc, UNOMax, NOK, 4) - self.MM(NOD, UNOMax, NOK, 4))                            
                    if  mcs % 360 == 0:  
                        output_dir = self.output_dir
                        if output_dir is not None:
                            output_path = Path(output_dir).joinpath('QEC' + '.csv')
                            with open(output_path, 'a+') as fout:
                                fout.write('{} {} {} {} \n'.format(mcs, cell.id, cell.dict["Damage"], cell.dict["Health"]))        
                     
                if cell.type == self.AEC:                   
                    #NO_conc = cell.dict['NewNO']                                                  
                    # Damage pathway
                    if NO_conc < NOD:
                        cell.dict["Damage"] += abs(self.MM(NO_conc, UNOMax, NOK, 4) - self.MM(NOD, UNOMax, NOK, 4))                                                   
                    # Healthy pathway
                    if NO_conc >= NOD:
                        cell.dict["Health"] += abs(self.MM(NO_conc, UNOMax, NOK, 4) - self.MM(NOD, UNOMax, NOK, 4))
                        print ("Health = ", cell.dict["Health"])
                    if  mcs % 360 == 0:     
                        output_dir = self.output_dir
                        if output_dir is not None:
                            output_path = Path(output_dir).joinpath('AEC' + '.csv')
                            with open(output_path, 'a+') as fout:
                                fout.write('{} {} {} {} \n'.format(mcs, cell.id, cell.dict["Damage"], cell.dict["Health"])) 
                  
class CellStateTransitionSteppable(SteppableBasePy):
    
    def start(self):
        # 初始化静息内皮细胞数量
        self.shared_steppable_vars['a'] = 0
        self.shared_steppable_vars['b'] = 0
        
    def step(self,mcs):
        
        for cell in self.cell_list:
            
            if  cell.type == self.QEC:                
                if  cell.dict["Damage"] >= 50:    #原来是60,120,180   
                    a1 = cell.dict['G'] 
                    a2 = cell.dict['IP3'] 
                    a3 = cell.dict['PIP2'] 
                    a4 = cell.dict['PIP3']
                    a5 = cell.dict['CAc']
                    a6 = cell.dict['CAs']
                    a7 = cell.dict['CAb']
                    a8 = cell.dict['AKTa'] 
                    a9 = cell.dict['PKCa']
                    a10 = cell.dict['Ca4CaM'] 
                    a11 = cell.dict['eNOS_CaM']
                    a12 = cell.dict['eNOS_CaMa']
                    a13 = cell.dict['eNOScav0']
                    a14 = cell.dict['NO']
                    a15 = cell.dict['cGMP']
                    a16 = cell.dict['eNOScav']
                    cell.type = self.AEC
                    cell.dict['G'] = a1 
                    cell.dict['IP3'] = a2
                    cell.dict['PIP2'] = a3
                    cell.dict['PIP3'] = a4
                    cell.dict['CAc'] = a5
                    cell.dict['CAs'] = a6
                    cell.dict['CAb'] = a7
                    cell.dict['AKTa'] = a8
                    cell.dict['PKCa'] = a9
                    cell.dict['Ca4CaM'] = a10
                    cell.dict['eNOS_CaM'] = a11
                    cell.dict['eNOS_CaMa'] = a12
                    cell.dict['eNOScav0'] = a13
                    cell.dict['NO'] = a14
                    cell.dict['cGMP'] = a15
                    cell.dict['eNOScav'] = a16
                    cell.dict["Health"] = 0 
                    cell.dict['current_mcs'] = 0
                    cell.dict["apoptosis"] = 0
                    cell.dict["ROS"] = 0
                    cell.targetVolume = 25
                    cell.lambdaVolume = 10.0  # 调节体积约束强度的参数
                    cell.targetSurface = ktgs*sqrt(cell.targetVolume)  
                    cell.lambdaSurface = 10.0   # 表面积大一点以维持形状   
                    
            if  cell.type == self.AEC:             
                if  cell.dict["apoptosis"]  >=1440:             
                    cell.type = self.DEC
                    cell.targetVolume = 25
                    cell.lambdaVolume = 10.0  # 调节体积约束强度的参数
                    cell.targetSurface = ktgs*sqrt(cell.targetVolume)  
                    cell.lambdaSurface = 10.0   # 表面积大一点以维持形状
                    self.shared_steppable_vars['a'] += 1
                if  cell.dict["Damage"] >= 400:  #原来是120, 分别选2倍, 120, 240, 360     
                    cell.type = self.DEC
                    cell.targetVolume = 25
                    cell.lambdaVolume = 10.0  # 调节体积约束强度的参数
                    cell.targetSurface = ktgs*sqrt(cell.targetVolume) 
                    cell.lambdaSurface = 10.0   # 表面积大一点以维持形状
                    self.shared_steppable_vars['b'] += 1
                if  cell.dict["Health"] >= 30:
                    a1 = cell.dict['G'] 
                    a2 = cell.dict['IP3'] 
                    a3 = cell.dict['PIP2'] 
                    a4 = cell.dict['PIP3']
                    a5 = cell.dict['CAc']
                    a6 = cell.dict['CAs']
                    a7 = cell.dict['CAb']
                    a8 = cell.dict['AKTa'] 
                    a9 = cell.dict['PKCa']
                    a10 = cell.dict['Ca4CaM'] 
                    a11 = cell.dict['eNOS_CaM']
                    a12 = cell.dict['eNOS_CaMa']
                    a13 = cell.dict['eNOScav0']
                    a14 = cell.dict['NewNO'] / 0.6
                    a15 = cell.dict['cGMP']
                    a16 = cell.dict['eNOScav']
                    cell.type = self.QEC
                    cell.dict['G'] = a1 
                    cell.dict['IP3'] = a2
                    cell.dict['PIP2'] = a3
                    cell.dict['PIP3'] = a4
                    cell.dict['CAc'] = a5
                    cell.dict['CAs'] = a6
                    cell.dict['CAb'] = a7
                    cell.dict['AKTa'] = a8
                    cell.dict['PKCa'] = a9
                    cell.dict['Ca4CaM'] = a10
                    cell.dict['eNOS_CaM'] = a11
                    cell.dict['eNOS_CaMa'] = a12
                    cell.dict['eNOScav0'] = a13
                    cell.dict['NO'] = a14
                    cell.dict['cGMP'] = a15
                    cell.dict['eNOScav'] = a16
                    cell.dict["Damage"] = 0
                    cell.dict["Health"] = 0
                    cell.targetVolume = 25
                    cell.lambdaVolume = 500.0  # 调节体积约束强度的参数
                    cell.targetSurface = ktgs*sqrt(cell.targetVolume) 
                    cell.lambdaSurface = 500.0   # 表面积大一点以维持形状  
                              
            if cell.type == self.MC:
                for neighbor, commonSurfaceArea in self.get_cell_neighbor_data_list(cell):
                    if neighbor:
                        if neighbor.type == self.AEC:
                            if(commonSurfaceArea) > (0.001) and cell.dict['Adhesiontime'] >= 30:
                                cell.yCOM -= 0.5
                
                x, y, z = cell.xCOM, cell.yCOM, cell.zCOM
                
                if  30 <= y <= 100:
                    if  cell.dict['lifespan'] >= 1440:
                        cell.type = self.DEAD
                    else:
                        cell.dict['lifespan'] +=1
                                          
                else:
                    temp = random.uniform(0,1)
                    if temp >=0.333:
                        cell.type = self.M1
                        cell.dict['lifespan'] = 0
                    else:
                        cell.type = self.M2
                        cell.dict['lifespan'] = 0

class MonocyteRecruitmentSteppable(SteppableBasePy):

    def start(self):
        # 初始化静息内皮细胞数量
        #self.qec_num = len(self.cell_list_by_type(self.QEC))
        self.has_recruited_first_time = False  # 标记是否已经首次招募
        self.last_recruit_mcs = 0  # 记录上次招募的时间
        self.MC_recruit_dict = {
            'total': 0,        # MC 累计招募总数
            'by_mcs': {},      # 每个 mcs 招募的 MC 数
            'events': 0        # 招募事件次数
        }
        self.shared_steppable_vars['MC_recruit_dict'] = self.MC_recruit_dict
        
    def step(self, mcs):
        if mcs > 550:
            # 初始化统计变量
            self.qec_num = len(self.cell_list_by_type(self.QEC))
            self.aec_num = len(self.cell_list_by_type(self.AEC))
            
            # 计算激活内皮细胞比例
            if self.qec_num > 0:  # 防止除零错误
                recruit_ratio = self.aec_num  / (self.qec_num + self.aec_num)
            else:
                recruit_ratio = 0

            # 判断是否满足招募条件
            if self.aec_num > 0:  # 存在激活内皮细胞
                # 首次出现AEC时招募一次
                if not self.has_recruited_first_time:
                    self.recruit_monocytes(recruit_ratio)
                    self.has_recruited_first_time = True
                    self.last_recruit_mcs = mcs
                # 之后的招募按照每120 MCS进行一次
                elif mcs - self.last_recruit_mcs >= 120:
                    self.recruit_monocytes(recruit_ratio)
                    self.last_recruit_mcs = mcs
                   
    def recruit_monocytes(self, recruit_ratio):
    
        # 动态计算招募数量
        if recruit_ratio >= 0.75:
            total_recruit = 3
        elif recruit_ratio >= 0.5:
            total_recruit = 2
        else:
            total_recruit = 1
    
        # ===== 统计 MC 招募 =====
        self.MC_recruit_dict['total'] += total_recruit
        self.MC_recruit_dict['events'] += 1
    
        mcs = self.simulator.getStep()
        if mcs not in self.MC_recruit_dict['by_mcs']:
            self.MC_recruit_dict['by_mcs'][mcs] = 0
        self.MC_recruit_dict['by_mcs'][mcs] += total_recruit
        # ==========================
    
        # 执行细胞招募
        for _ in range(total_recruit):
            cell = self.new_cell(self.MC)
            x = random.randint(0, self.dim.x - 5)
            y = random.randint(45, self.dim.y - 5)
            self.cell_field[x:x + 5, y:y + 5, 0] = cell
    
            cell.targetVolume = 36
            cell.lambdaVolume = 15
            cell.targetSurface = ktgs * sqrt(cell.targetVolume)
            cell.lambdaSurface = 5
            cell.dict['lifespan'] = 0
            cell.dict['Adhesiontime'] = 0


class MitosisSteppable(MitosisSteppableBase):
    def _init_(self, frequency = 1):
        MitosisSteppableBase._init_(self, frequency)
        
    def step(self,mcs):      
        cells_to_divide=[] 
        for cell in self.cell_list:                
            if (cell.type == self.AEC and cell.volume >= 50 and cell.yCOM <= 40):
                cells_to_divide.append(cell)
        for cell in cells_to_divide:           
            self.divide_cell_random_orientation(cell)
 
    def update_attributes(self):
        # print("12223")
        self.parent_cell.targetVolume=25  
        self.child_cell.targetVolume=25
        self.parent_cell.targetSurface= ktgs*sqrt(self.parent_cell.targetVolume) 
        self.child_cell.targetSurface= ktgs*sqrt(self.child_cell.targetVolume)
        self.parent_cell.lambdaVolume= 10
        self.parent_cell.lambdaSurface= 10
        self.child_cell.lambdaVolume= 10
        self.child_cell.lambdaSurface= 10
        
        if self.parent_cell.type == self.AEC: 
        
            #temp=random.gauss(maxdiv,2)
            #if  self.parent_cell.dict["Counter"] <= temp :
            self.parent_cell.type=self.AEC            
            self.child_cell.type=self.AEC
                #self.parent_cell.dict["Counter"]+=1
                               
            self.clone_parent_2_child()
            
            #if self.parent_cell.dict["Counter"] > temp :
               #self.parent_cell.type=self.DEC
               #self.child_cell.type=self.DEC
                            
        #self.parent_cell.dict["Damage"] = 0
        self.child_cell.dict["Damage"] = 0
        #self.parent_cell.dict["Health"] = 0
        self.child_cell.dict["Health"] = 0
        #self.parent_cell.dict["current_mcs"] = 0
        self.child_cell.dict["current_mcs"] = 0

class DataOutputSteppable(SteppableBasePy): 
         
    def step(self,mcs): 
        
        QEC = len(self.cell_list_by_type(self.QEC))
        AEC = len(self.cell_list_by_type(self.AEC))
        DEC = len(self.cell_list_by_type(self.DEC))
        if 'MC_recruit_dict' not in self.shared_steppable_vars:
            return
        MC = self.shared_steppable_vars['MC_recruit_dict']['total']
        M1 = len(self.cell_list_by_type(self.M1))
        M2 = len(self.cell_list_by_type(self.M2))
        DEAD = len(self.cell_list_by_type(self.DEAD))
        count1 = self.shared_steppable_vars['a'] #坏死
        count2 = self.shared_steppable_vars['b'] #Damage
        
        RQEC = QEC/(QEC+AEC+count1+count2)
        RAEC = AEC/(QEC+AEC+count1+count2)
        RDEC = (count1+count2)/(QEC+AEC+count1+count2)
        
        if MC > 0:
            RM1 = M1/MC
            RM2 = M2/MC
            RDMC = (MC-M1-M2)/MC
        else: 
            RM1 = math.nan
            RM2 = math.nan
            RDMC = math.nan
        output_dir = self.output_dir
        if output_dir is not None:
            output_path = Path(output_dir).joinpath('cellnumber' + '.csv')
            with open(output_path, 'a+') as fout:
                fout.write('{} {} {} {} {} {} {} {} {} {}\n'.format(mcs,QEC,AEC,DEC,MC,M1,M2,DEAD, count1, count2))        
        
        output_dir = self.output_dir
        if output_dir is not None:
            output_path = Path(output_dir).joinpath('cellratio' + '.csv')
            with open(output_path, 'a+') as fout:
                fout.write('{} {} {} {} {} {} {} \n'.format(mcs,RQEC,RAEC,RDEC,RM1,RM2,RDMC))  
           