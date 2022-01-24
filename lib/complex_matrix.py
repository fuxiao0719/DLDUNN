import torch
import numpy as np

def c2m(A):
    Cre = torch.from_numpy(np.real(A))
    Cim = torch.from_numpy(np.imag(A))
    return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)

def mcat(A,B):
    Cre = A
    Cim = B
    return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)

def cmul(A,B):
    Cre = torch.mm(A[0,:],B[0,:]) - torch.mm(A[1,:],B[1,:])
    Cim = torch.mm(A[0,:],B[1,:]) + torch.mm(A[1,:],B[0,:])
    return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)

def cmul1(a,B):
    Cre = a[0] * B[0, :] - a[1] * B[1, :]
    Cim = a[0] * B[1, :] + a[1] * B[0, :]
    return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)

def cmul2(a,b):
    cre = a[0] * b[0] - a[1] * b[1]
    cim = a[0] * b[1] + a[1] * b[0]
    return torch.cat((cre.unsqueeze(0),cim.unsqueeze(0)),0)

def conjT(A):
    Bre = torch.transpose(A[0,:],0,1)
    Bim = torch.transpose(-A[1,:],0,1)
    return torch.cat((Bre.unsqueeze(0),Bim.unsqueeze(0)),0)

def conj(a):
    bre = a[0]
    bim = -a[1]
    return torch.cat((bre.unsqueeze(0),bim.unsqueeze(0)),0)

def cpinv(A):
    return cmul(cinv(cmul(conjT(A), A)), conjT(A))

def cinv(A):
    try:
        Cre = torch.inverse(A[0,:]+torch.mm(A[1,:],torch.mm(torch.inverse(A[0,:]),A[1,:])))
        Cim = -torch.mm(torch.mm(torch.inverse(A[0,:]),A[1,:]),Cre)
    except:
        Cre = torch.pinverse(A[0,:]+torch.mm(A[1,:],torch.mm(torch.pinverse(A[0,:]),A[1,:])))
        Cim = -torch.mm(torch.mm(torch.pinverse(A[0,:]),A[1,:]),Cre)
    return torch.cat((Cre.unsqueeze(0), Cim.unsqueeze(0)), 0)

def cdiag(a):
    Are = torch.diag(a[0,:])
    Aim = torch.diag(a[1,:])
    return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)


def cdiv(A):
    Are = torch.diag(torch.div(torch.diag(A[0,:]),torch.diag(A[0,:])**2+torch.diag(A[1,:])**2))
    Aim = torch.diag(torch.div(-torch.diag(A[1,:]),torch.diag(A[0,:])**2+torch.diag(A[1,:])**2))
    return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)

def cdiv1(B,a):
    de = (a[0]**2+a[1]**2)
    Are = a[0]/de * B[0,:] + a[1]/de * B[1,:]
    Aim = a[0]/de * B[1,:] - a[1]/de * B[0,:]
    return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)

def c1div(a):
    de = (a[0]**2+a[1]**2)
    Are = a[0]/de
    Aim = -a[1]/de
    return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)

def cdiv2(a,b):
    return cmul2(a,c1div(b))

def cdet(A):
    return A[0,0,0]*A[0,1,1]-A[1,1,1]*A[1,0,0]-A[0,1,0]*A[0,0,1]+A[1,1,0]*A[1,0,1] 

def laplace_temp(A, idx, jdx):
    n = A.size(1)-1
    A_real = torch.zeros((n,n))
    A_imag = torch.zeros((n,n))
    index_x = 0
    index_y = 0
    for i in range(n+1):
        index_y = 0
        for j in range(n+1):        
            if (i!=idx)&(j!=jdx):
                A_real[index_x,index_y] = A[0,i,j]
                A_imag[index_x,index_y] = A[1,i,j]
                index_y += 1
        if (i!=idx):
            index_x += 1
    return torch.cat((A_real.unsqueeze(0),A_imag.unsqueeze(0)),0)

def complex_det(A):
    n = A.size(1)
    if n == 1:
        return torch.cat((A[0,0,0].unsqueeze(0),A[1,0,0].unsqueeze(0)),0)
    else: 
        det = torch.tensor([0., 0.])
        for j in range(n):
            det += (-1)**((2+j)%2)*cmul2(mcat(A[0,0,j],A[1,0,j]),complex_det(laplace_temp(A,0,j)))
        return det