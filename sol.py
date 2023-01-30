from os import R_OK
from re import A
import sys
import json
import dynamical_system
import ds_func
import dp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import sin,cos


# # プロット用設定
# plt.rcParams["font.family"] = "Nimbus Roman"    #全体のフォントを設定
# plt.rcParams['text.usetex'] = True              #描画にTeXを利用
# plt.rcParams['text.latex.preamble'] = r'''\usepackage{amsmath}
#                                           \usepackage{amssymb}
#                                           \usepackage[T1]{fontenc}
#                                           \usepackage{bm}
#                                           \usepackage{xcolor}
#                                           '''
# plt.rcParams["figure.autolayout"] = False       #レイアウト自動調整をするかどうか
# plt.rcParams["font.size"] = 24                  #フォントの大きさ
# plt.rcParams["xtick.direction"] = "in"          #x軸の目盛線を内向きへ
# plt.rcParams["ytick.direction"] = "in"          #y軸の目盛線を内向きへ
# plt.rcParams["xtick.minor.visible"] = True      #x軸補助目盛りの追加
# plt.rcParams["ytick.minor.visible"] = True      #y軸補助目盛りの追加
# plt.rcParams["xtick.major.width"] = 1.0         #x軸主目盛り線の線幅
# plt.rcParams["ytick.major.width"] = 1.0         #y軸主目盛り線の線幅
# plt.rcParams["xtick.minor.width"] = 0.5         #x軸補助目盛り線の線幅
# plt.rcParams["ytick.minor.width"] = 0.5         #y軸補助目盛り線の線幅
# plt.rcParams["xtick.major.size"] = 20           #x軸主目盛り線の長さ
# plt.rcParams["ytick.major.size"] = 20          #y軸主目盛り線の長さ
# plt.rcParams["xtick.minor.size"] = 10            #x軸補助目盛り線の長さ
# plt.rcParams["ytick.minor.size"] = 10            #y軸補助目盛り線の長さ
# plt.rcParams["xtick.major.pad"] = 16             #x軸と目盛数値のマージン
# plt.rcParams["ytick.major.pad"] = 16             #y軸と目盛数値のマージン
# plt.rcParams["axes.linewidth"] = 2            #囲みの太さ


def func(t, x, p, c):
    M1 = c[0]
    M2 = c[1]
    L1 = c[2]
    L2 = c[3]
    G = c[4]


    a11 = M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12 = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21 = a12
    a22 = M2 * L2 * L2
    b1 = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2 = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    delta = a11 * a22 - a12 * a21

    ret = np.array([
        x[1],
        (b1 * a22 - b2 * a12) / delta,
        x[3],
        (b2 * a11 - b1 * a21) / delta
    ])
    return ret

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)
    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    fd.close()
    # input data to constructor
    ds = dynamical_system.DynamicalSystem(json_data)
    
    ##################################################
    # solve
    ##################################################
    tau1= json_data['tau1']
    tau2= json_data['tau2']
    duration = [0,tau1] 
    duration_m = [0,tau2]
    tick = 0.01
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    ds.c = ds_func.sp2np(ds.const).flatten()
    # calculate equilibrium points
    spp = sp.Matrix(p)
    ds.sp_x0 = Eq_check(spp,ds)
    print("sp_x0",ds.sp_x0)
    stable = ds_func.set_x0(p,ds.c)
    stable = ds_func.sp2np(stable)
    stable = stable[3,:]
    print("stable",stable)
    ds.x0 = ds_func.sp2np(ds.sp_x0)
    Eigen(ds.sp_x0,spp,ds)

    ds.state0 = ds.x0
    print(ds.state0)
    x_alpha = ds.x_alpha
    x_omega = ds.x_omega
    state = solve_ivp(func, duration, ds.state0.flatten(),
        method='RK45', args = (p, ds.c),
        rtol=1e-12)
    state_p = solve_ivp(func, duration, x_alpha,
        method='RK45', args = (p, ds.c),
        rtol=1e-12, atol = 1e-9)
    # print("t-p",state_p.t)
    # print("y-p",state_p.y)
    state_m = solve_ivp(func, duration_m, x_omega,
        method='RK45',args = (p, ds.c),
        rtol=1e-12, atol = 1e-9)
    
    fig = plt.figure(figsize = (16, 8))
    ax1 = fig.add_subplot(221)
    # label = "saddle"
    # color = (0.0, 1.0, 0.0)
    # ax1.plot(state.y[0,:], state.y[1,:],
    #         linewidth=1, color = color,
    #         label = label,ls="-")
    label = "unstable"
    color = (1.0, 0.0, 0.0)
    ax1.plot(state_p.y[0,:], state_p.y[1,:],
            linewidth=1, color = color,
            label = label,ls="-")


    color = (0.0, 0.0, 1.0)
    label = "stable"
    ax1.plot(state_m.y[0,:], state_m.y[1,:],
            linewidth=1, color = color,
            label = label,ls="-")
    color = (0.0, 0.0, 0.0)
    print("alpha",state_p.y[0,-1])
    print("omega",state_m.y[0,-1])
    print("saddle",state.y[0,-1])
    ax1.plot(ds.state0[0,0],0,".",markersize=2,color="blue")
    ax1.plot(ds.state0[0,0]+2*np.pi,0,".",markersize=2,color="blue")
    ax1.plot(ds.state0[0,0]-2*np.pi,0,".",markersize=2,color="blue")
    # ax1.plot(ds.state0[0,0]+4*np.pi,0,".",markersize=2,color="blue")
    # ax1.plot(ds.state0[0,0]-4*np.pi,0,".",markersize=2,color="blue")
    # plt.plot(-np.pi/2+2*np.pi,0,"o",markersize=15,color="red")
    ax1.plot(stable[0]+2*np.pi,0,".",markersize=2,color="red")
    ax1.plot(stable[0],0,".",markersize=2,color="red")
    # show_param(ds)
    # plt.show()
    ax2 = fig.add_subplot(222)
    label = "unstable"
    color = (1.0, 0.0, 0.0)
    ax2.plot(state_p.y[2,:], state_p.y[3,:],
            linewidth=1, color = color,
            label = label,ls="-")
    
    color = (0.0, 0.0, 1.0)
    label = "stable"
    ax2.plot(state_m.y[2,:], state_m.y[3,:],
            linewidth=1, color = color,
            label = label,ls="-")

    ax2.plot(ds.state0[2,0],0,".",markersize=2,color="blue")
    ax2.plot(ds.state0[2,0]+2*np.pi,0,".",markersize=2,color="blue")
    ax2.plot(ds.state0[2,0]-2*np.pi,0,".",markersize=2,color="blue")
    ax2.plot(ds.state0[2,0]+4*np.pi,0,".",markersize=2,color="blue")
    ax2.plot(ds.state0[2,0]-4*np.pi,0,".",markersize=2,color="blue")
    # plt.plot(-np.pi/2+2*np.pi,0,"o",markersize=15,color="red")
    # ax2.plot(stable[2]+2*np.pi,0,".",markersize=2,color="red")
    # ax2.plot(stable[2],0,".",markersize=2,color="red")

    ax3 = fig.add_subplot(223,projection='polar') 
    
    ax3.plot(state_p.y[0],state_p.y[1],color='red')
    ax3.plot(state_m.y[0],state_m.y[1],color='blue')
    # ax.plot(solve_saddle_minus.y[0],solve_saddle_minus.y[1],color='black')
    # ax.plot(solve_alpha.y[0],solve_alpha.y[1],color = 'black')
    # ax.plot(solve_omega.y[0],solve_omega.y[1],color = 'black')
    # r方向の設定
    # ax.set_rticks((-6,-4,-2,0,2,4,6))
    # ax3.set_rticks((0,1,2))
    ax3.set_rlabel_position(0)

    # 回転軸の設定
    ax3.set_thetalim([0,2 * np.pi])
    ax3.set_thetagrids(np.rad2deg(np.linspace(0, 2*np.pi,9)[1:]), 
                    labels=[ "π/4", "π/2", "3π/4", "π","5π/4","3π/2","7π/4","0"], 
    #                   labels=[r'$\displaystyle\frac{\pi}{4}$', r'$\displaystyle\frac{\pi}{2}$', r'$\displaystyle\frac{3\pi}{4}$', r'$\displaystyle\pi$',r'$\displaystyle\frac{5\pi}{4}$',r'$\displaystyle\frac{3\pi}{2}$',r'$\displaystyle\frac{7\pi}{4}$',r'0'], 
                      fontsize=12)
    # 外枠の軸を消す                 
    ax3.spines['polar'].set_visible(False)  
    ax4 = fig.add_subplot(224,projection='polar') 
    ax4.plot(state_p.y[2],state_p.y[3],color='red')
    ax4.plot(state_m.y[2],state_m.y[3],color='blue')
    # ax.plot(solve_saddle_minus.y[0],solve_saddle_minus.y[1],color='black')
    # ax.plot(solve_alpha.y[0],solve_alpha.y[1],color = 'black')
    # ax.plot(solve_omega.y[0],solve_omega.y[1],color = 'black')
    # r方向の設定
    # ax.set_rticks((-6,-4,-2,0,2,4,6))
    # ax4.set_rticks((0,1,2))
    ax4.set_rlabel_position(0)

    # 回転軸の設定
    ax4.set_thetalim([0,2 * np.pi])
    ax4.set_thetagrids(np.rad2deg(np.linspace(0, 2*np.pi,9)[1:]), 
                    labels=[ "π/4", "π/2", "3π/4", "π","5π/4","3π/2","7π/4","0"], 
    #                   labels=[r'$\displaystyle\frac{\pi}{4}$', r'$\displaystyle\frac{\pi}{2}$', r'$\displaystyle\frac{3\pi}{4}$', r'$\displaystyle\pi$',r'$\displaystyle\frac{5\pi}{4}$',r'$\displaystyle\frac{3\pi}{2}$',r'$\displaystyle\frac{7\pi}{4}$',r'0'], 
                      fontsize=12)
    # 外枠の軸を消す                 
    ax4.spines['polar'].set_visible(False)  
    plt.show()



def show_param(ds):
    s = ""
    p = ""
    params = ds_func.sp2np(ds.params).flatten().tolist()
    x0 = ds.state0.flatten().tolist()
    for i in range(len(params)):
        s += f"x{i}:{x0[i]:.5f},"
        p += f"p{i}:{params[i]:.4f},"
    plt.title(s+"\n"+p, color = 'blue')

def Eigen(x0, p, ds):
    #パラメータに依存するように
    print("for eigen",x0)
    print("for eigen",p)
    eig,eig_vl,eig_vr = ds_func.eigen(x0, p, ds)
    print("eigenvalue\n", eig)
    print("eigen_vector\n", eig_vr)

    ds.mu_alpha = eig[1].real
    ds.mu_omega = eig[0].real
    # eig_vr = eig_vr * (-1)
    # print("eigen_vector",*eig_vr[:,].T,sep='\n')
    # eig_vr = eig_vr * (-1)
    delta_alpha = eig_vr * ds.delta1
    print("delta",delta_alpha)
    np_x0 = ds_func.sp2np(x0)
    print("x0",np_x0)
    ####ここは８本から２本選択
    ds.x_alpha = (np_x0[:,0] + delta_alpha[:,3]).flatten().real
    delta_omega = eig_vr * ds.delta2
    ds.x_omega = (np_x0[:,0] + delta_omega[:,1]).flatten().real
    print("mu_alpha", ds.mu_alpha)
    print("mu_omega", ds.mu_omega)
    print("x_alpha", ds.x_alpha)
    print("x_omega", ds.x_omega)

def Eq_check(p,ds):
    eq = ds_func.set_x0(p,ds.c)
    vp = eq[0,:].T
    for i in range(ds.ite_max):
        F = ds.F.subs([(ds.sym_x, vp), (ds.sym_p, p)])
        J = ds.dFdx.subs([(ds.sym_x, vp), (ds.sym_p, p)])
        F = ds_func.sp2np(F)
        J = ds_func.sp2np(J)
        dif = abs(np.linalg.norm(F))
        # print("dif=",dif)
        if dif < ds.eps:
            # print("success!!!")
            print("solve vp = ",vp)
            return vp
        
        if dif > ds.explode:
            print("Exploded")
            exit()
            # vn = xk+1
            # print("vp=",vp)
        vn = np.linalg.solve(J,-F) + vp
        print("vn=",vn)
        vp = vn
        # if vn[5] > 1.0:
        #   print("B0 is too high")




if __name__ == '__main__':
    main()