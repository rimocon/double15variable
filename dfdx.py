import sympy as sp
import numpy as np
# from numpy import cos,sin
from sympy import cos,sin
import json

def map(x, p, c):

    # double pendulum  equation
    # x:variable, p:parameters, c:constant

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

    ret = sp.Matrix([x[1],
                    (b1 * a22 - b2 * a12) / delta,
                    x[3],
                    (b2 * a11 - b1 * a21) / delta
                    ])
    return ret

sym_x = sp.MatrixSymbol('x', 4, 1)
sym_p = sp.MatrixSymbol('p', 4, 1)
c = sp.Matrix([
        1.0,
        1.0,
        1.0,
        1.0,
        9.8066500000
    ])
F = map(sym_x, sym_p, c)
print("F", F)
dFdx = F.jacobian(sym_x)
print("dFdx", dFdx)

print("dFdx 1 gyoume", dFdx[1,:])
a = str(dFdx[0,:])
b = str(dFdx[1,:])
c = str(dFdx[2,:])
d = str(dFdx[3,:])
with open('dfdx.json', 'w') as f:
    json.dump(a, f)
    json.dump(b, f)
    json.dump(c, f)
    json.dump(d, f)