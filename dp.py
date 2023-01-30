from re import A
import sys
import json
import dynamical_system
import ds_func
# import variation
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import sin,cos
from scipy import linalg


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
    # 初期値に関する変分方程式
    dFdx = np.array([[0, 1, 0, 0],
    # 2行目
    [(9.80665*(-1.0*cos(x[2]) - 1.0)*sin(x[0] + x[2]) + 9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0) ,
    (-2.0*(-1.0*cos(x[2]) - 1.0)*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3] - 1.0*p[0])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    0.111111111111111*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))*(-(1.0*cos(x[2]) + 1.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]) + 2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - 1.0*p[0]*x[1] + 1.0*p[2])/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(-1.0*cos(x[2]) - 1.0) + 1.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (-(-1.0*cos(x[2]) - 1.0)*p[1] + 2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
    [0, 0, 0, 1],
    # ４行目
    [((9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))*(-1.0*cos(x[2]) - 1.0) + 9.80665*(2.0*cos(x[2]) + 3.0)*sin(x[0] + x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    ((2.0*sin(x[2])*x[3] - p[0])*(-1.0*cos(x[2]) - 1.0) - 2.0*(2.0*cos(x[2]) + 3.0)*sin(x[2])*x[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    0.111111111111111*(-(1.0*cos(x[2]) + 1.0)*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2]) + (2.0*cos(x[2]) + 3.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]))*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(2.0*cos(x[2]) + 3.0) + (-1.0*cos(x[2]) - 1.0)*(9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2) - 2.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 1.0*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2])*sin(x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    ((2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])*(-1.0*cos(x[2]) - 1.0) - (2.0*cos(x[2]) + 3.0)*p[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
    ])
    dphidx = x[4:20].reshape(4,4)
    # print("dphidx",dphidx)
    # print("dFdx",dFdx)
    i = (dFdx @ dphidx).flatten()
    # ## パラメタに関する変分方程式
    dFdl = np.array([[0, 0, 0, 0],
    # 2行目
    [-1.0*x[1]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    -(-1.0*cos(x[2]) - 1.0)*x[3]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    1.0/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (-1.0*cos(x[2]) - 1.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
    [0, 0 ,0, 0],
    # 4行目
    [-(-1.0*cos(x[2]) - 1.0)*x[1]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    -(2.0*cos(x[2]) + 3.0)*x[3]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    (-1.0*cos(x[2]) - 1.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (2.0*cos(x[2]) + 3.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
    ])
    # print("dFdl",dFdl[:,3])
    ##この4本から1本だけでいい
    dphidl = np.array(x[20:24])
    # ここの:,1←ここは選択するパラメタによって変える
    p = (dFdx@ dphidl.reshape(4,1) + dFdl[:,2].reshape(4,1)).flatten()
    ## 元の微分方程式+変分方程式に結合
    ret = np.block([ret,i,p])
    return ret 

def main():
    # load data from json file
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)

    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    fd.close()

    # input data to constructor
    ds = dynamical_system.DynamicalSystem(json_data)
    # import numpy parameter
    ds.p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    ds.c = ds_func.sp2np(ds.const).flatten()

    # ここで平衡点+固有値などセット
    spp = sp.Matrix(ds.p)
    ds.sp_x0 = Eq_check(spp,ds)
    ds.x0 = ds_func.sp2np(ds.sp_x0)
    x0 = ds.x0.flatten()
    Eigen(ds.sp_x0,spp,ds)


    vp = np.block([x0, ds.x_alpha, ds.x_omega,ds.duration[1],ds.duration_m[1],ds.p[ds.var]])

    dfdl = ds.F.jacobian(ds.sym_p)
    print("dfdl",dfdl)
    # solve(vp,ds)
    # F = ds_func.F(vp, ds.p, ds)
    # J = ds_func.J(vp, ds.p, ds)
    # print("F",F)
    # print("F=",F)
    # print("J=",J)
    # z = np.block([vp[0:4],vp[4:8],vp[8:12],ds.p])
    # z = sp.Matrix(z.reshape(16,1))
    # dfdx = ds.dFdx_n.subs(ds.sym_z, z)
    # print("sympy",dfdx)
    ds.sym_F = ds_func.newton_F(ds.sym_z,ds)
    # # print("sym_F= ",ds.sym_F)
    ds.sym_J = ds.sym_F.jacobian(ds.sym_z)
    # # print("sym_J= ",ds.sym_J)
    # # J = ds.sym_J.subs(ds.sym_z,z)
    # # print(J)
    # # print("sym_J= ",ds.sym_J.shape)
    newton_method(vp,ds)
   

def Condition(z,ds):
    F = ds.sym_F.subs(ds.sym_z,z)
    J = ds.sym_J.subs(ds.sym_z,z)
    F = ds_func.sp2np(F)
    J = ds_func.sp2np(J)
    homo = ds.state_p.y[0:4,-1] - ds.state_m.y[0:4,-1]
    
    F[11] = homo[0]
    # F[11] = homo[0] - np.pi
    # if F[10] > 2*np.pi:
    #     F[10] - 2*np.pi
    # while np.abs(F[11]) > 2* np.pi:
    #     if F[11] > 2*np.pi:
    #         F[11] = F[11] - 2*np.pi
    #         print("homo - 2*np.pi")
    #     elif F[11] < -2*np.pi:
    #         F[11] = F[11] + 2*np.pi
    #         print("homo + 2*np.pi")
    #     else:
    #         print("break")
    #         break
    F[12] = homo[1] 
    F[13] = homo[2] 
    # while np.abs(F[13]) > 2* np.pi:
    #     if F[13] > 2*np.pi:
    #         F[13] = F[13] - 2*np.pi
    #         print("homo - 2*np.pi")
    #     elif F[13] < -2*np.pi:
    #         F[13] = F[13] + 2*np.pi
    #         print("homo + 2*np.pi")
    #     else:
    #         print("break")
     
    F[14] = homo[3]
    print("F subs",F)
    # print("J subs",J)
    dFdlambda = J[:,12+ds.var].reshape(15,1)
    # print("lambda",J[:,12:16])
    # print("lambda",J[:,12+ds.var])
    dFdtau_1 = np.zeros((15,1))
    dFdtau_2 = np.zeros((15,1))
    J = np.block([[J[:,0:12],dFdtau_1,dFdtau_2,dFdlambda]])

    dphidxalpha = ds.state_p.y[4:20,-1].reshape((4,4)).T
    dphidxomega = ds.state_m.y[4:20,-1].reshape((4,4)).T
    dphidlambda = (ds.state_p.y[20:24,-1] - ds.state_m.y[20:24,-1])
    print("dphidl",dphidlambda)
    J[11:15,4:8] = dphidxalpha
    J[11:15,8:12] = -dphidxomega
    J[11:15,14] = dphidlambda

    p = ds.p
    p[ds.var] = z[12+ds.var]

    x = []
    x[0:4] = ds.state_p.y[0:4,-1]
    ret_p = Phi(x,p,ds)
    print("ret_p",ret_p)
    x[0:4] = ds.state_m.y[0:4,-1]
    ret_m = Phi(x,p,ds)
    print("ret_m",ret_m)
    ### ここプラスであってる？
    J[11][12] = ret_p[0]
    J[12][12] = ret_p[1]
    J[13][12] = ret_p[2]
    J[14][12] = ret_p[3]

    J[11][13] = ret_m[0]
    J[12][13] = ret_m[1]
    J[13][13] = ret_m[2]
    J[14][13] = ret_m[3]
    # J[10][12] = ret_p[0] + ret_m[0]
    # J[11][12] = ret_p[1] + ret_m[1]
    # J[12][12] = ret_p[2] + ret_m[2]
    # J[13][12] = ret_p[3] + ret_m[3]
    # J[8][9] = ds.state_p.y[1,-1] + ds.state_m.y[1,-1]
    # J[9][9] = ((b1p * a22p - b2p * a12p) / deltap) + ((b1m * a22m - b2m * a12m) / deltam)
    # print("J tau_1",J[:,12])
    # print("J tau_2",J[:,13])
    return F,J


def Eigen(x0, p, ds):
    #パラメータに依存するように
    eig,eig_vl,eig_vr= ds_func.eigen(x0, p, ds)
    print("eigenvalue\n", eig)
    print("eigen_vector\n", eig_vr)

    ds.mu_alpha = eig[3].real
    ds.mu_omega = eig[1].real
    delta1 = eig_vr * ds.delta1
    delta2 = eig_vr * ds.delta2
    np_x0 = ds_func.sp2np(x0)
    print("x0",np_x0)
    ####ここは８本から２本選択
    ds.x_alpha = (np_x0[:,0] + delta1[:,3]).flatten()
    ds.x_omega = (np_x0[:,0] + delta2[:,1]).flatten()
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
def Phi(x,p,ds):
    M1 = ds.c[0]
    M2 = ds.c[1]
    L1 = ds.c[2]
    L2 = ds.c[3]
    G = ds.c[4]

    
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
 

def solve(vp,ds):
    ####solve####
    # import numpy parameter
    # パラメータ
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    p[ds.var] = vp[14]
    x0_p = np.block([vp[4:8],ds.vari_ini])
    x0_m = np.block([vp[8:12],ds.vari_ini])
    print("initial value for plus",x0_p)
    print("initial value for minus",x0_m)
    # for plus
    ds.state_p = solve_ivp(func, [0,vp[12]], x0_p,
        method='RK45', args = (p, c),
        rtol=1e-12,atol=1e-5)
    # print("y_p",ds.state_p.y)
    ## for minus
    ds.state_m = solve_ivp(func, [0,vp[13]], x0_m,
        method='RK45', args = (p, c), 
        rtol=1e-12,atol=1e-5)
    # print("t_m",ds.state_m.t)
    # print("y_m",ds.state_m.y)

# ニュートン法
def newton_method(vp,ds):
    p = ds.p
    for i in range(ds.ite_max):
        print(f"###################iteration:{i}#######################")
        # パラメタだけニュートンで求まったものをセット
        p[ds.var] = vp[14]
        # # パラメタによってmu_alpha,mu_omegaが変化するのでセットし直す
        spp = sp.Matrix(p)
        ds.sp_x0 = sp.Matrix(vp[0:4])
        Eigen(ds.sp_x0,spp,ds)
        
        print("vp",vp)
        print("x0",vp[0:4])
        print("x_alpha",vp[4:8])
        print("x_omega",vp[8:12])
        print("tau_1",vp[12])
        print("tau_2",vp[13])
        print("lambda",vp[14])

        # 微分方程式+変分方程式を初期値vpで解く
        solve(vp,ds)
        z = np.block([vp[0:4],vp[4:8],vp[8:12],p])
        print("z=",z)
        z = sp.Matrix(z.reshape(16,1))
        ################################3
        # G = ds_func.F(vp, p, ds)
        # J = ds_func.J(vp, p, ds)
        F,J = Condition(z,ds)
        print("F",F)
        print("J",J)
        dif = abs(np.linalg.norm(F))
        print("diff=",dif)
        if dif < ds.eps:
            print("success!!!")
            print("solve vp = ",vp)
            return vp
        if dif > ds.explode:
            print("Exploded")
            exit()
        test = np.linalg.solve(J,-F)
        print("solve(J,-F) = ",test)
        vn = np.linalg.solve(J,-F).flatten() + vp
        print("vn=",vn)
        vp = vn
        

if __name__ == '__main__':
    main()