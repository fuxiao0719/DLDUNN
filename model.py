import torch
import numpy as np 
import torch.nn as nn
from complex_matrix import *

# the whole model
class model(nn.Module):
    def __init__(self,K,d,M,N,N_RF,M_RF,P,sigma,num_in,num_out,device):
        super(model,self).__init__()
        self.K = K
        self.d = d
        self.M = M 
        self.N = N 
        self.P = P
        self.N_RF = N_RF
        self.M_RF = M_RF
        self.sigma = sigma
        self.num_in = num_in
        self.num_out = num_out
        self.device = device
        self.Outer_loop = {}
        self.rho = nn.Parameter(torch.tensor(self.P/self.N*0.8))
        for i in range(self.num_out):
            self.Outer_loop['%d'%i] = Outer_layer(self.K,self.d,self.M,self.N,self.N_RF,self.M_RF,self.P,self.sigma,self.num_in)        #待定
        self.Outer_loop = nn.ModuleDict(self.Outer_loop)   

    def forward(self,H,U_BB,U_RF,V_BB,V_RF,W,X,Y): 
        U_BB_GROUP = {}
        U_RF_GROUP = {}
        V_BB_GROUP = {}
        V_RF_GROUP = {}
        W_GROUP = {}
        X_GROUP = {}
        Y_GROUP = {}
        U_BB_GROUP['%d'%0],U_RF_GROUP['%d'%0],V_BB_GROUP['%d'%0],V_RF_GROUP['%d'%0],\
                W_GROUP['%d'%0],X_GROUP['%d'%0],Y_GROUP['%d'%0] = U_BB,U_RF,V_BB,V_RF,W,X,Y
        for i in range(self.num_out):
            U_BB_GROUP['%d'%(i+1)],V_BB_GROUP['%d'%(i+1)],V_RF_GROUP['%d'%(i+1)],U_RF_GROUP['%d'%(i+1)]\
                ,W_GROUP['%d'%(i+1)],X_GROUP['%d'%(i+1)],Y_GROUP['%d'%(i+1)]= self.Outer_loop['%d'%i](H,\
                     U_BB_GROUP['%d'%i],U_RF_GROUP['%d'%i],V_BB_GROUP['%d'%i],V_RF_GROUP['%d'%i],W_GROUP['%d'%i],X_GROUP['%d'%i],Y_GROUP['%d'%i],self.rho)
        sum_temp = 0
        for k in range(self.K):
            sum_temp += torch.trace(cmul(conjT(cmul(V_RF_GROUP['%d'%(self.num_out)],V_BB_GROUP['%d'%(self.num_out)][k])),cmul(V_RF_GROUP['%d'%(self.num_out)],V_BB_GROUP['%d'%(self.num_out)][k]))[0])
        V_BB_GROUP['%d'%(self.num_out)] = V_BB_GROUP['%d'%(self.num_out)] * torch.sqrt(self.P / sum_temp)
        return U_BB_GROUP['%d'%(self.num_out)],U_RF_GROUP['%d'%(self.num_out)], V_BB_GROUP['%d'%(self.num_out)],V_RF_GROUP['%d'%(self.num_out)]                                   

class Outer_layer(nn.Module):
    def __init__(self,K,d,M,N,N_RF,M_RF,P,sigma,num_in):
        super(Outer_layer,self).__init__()
        self.K = K
        self.d = d
        self.M = M 
        self.N = N 
        self.P = P
        self.N_RF = N_RF
        self.M_RF = M_RF
        self.sigma = sigma
        self.num_in = num_in
        self.Y_f = Y_layer(self.K,self.N,self.d)
        self.Inner_loop = {}
        for i in range(self.num_in):
            self.Inner_loop['%d'%i] = Inner_layer(self.K,self.d,self.M,self.N,self.N_RF,self.M_RF,self.P,self.sigma)        #待定
        self.Inner_loop = nn.ModuleDict(self.Inner_loop)

    def forward(self,H,U_BB,U_RF,V_BB,V_RF,W,X,Y,rho):
        U_BB_GROUP = {}
        U_RF_GROUP = {}
        V_BB_GROUP = {}
        V_RF_GROUP = {}
        W_GROUP = {}
        X_GROUP = {}
        Y_new = torch.zeros(self.K,2,self.N,self.d)
        U_BB_GROUP["%d"%0],V_BB_GROUP["%d"%0],W_GROUP["%d"%0],X_GROUP["%d"%0],V_RF_GROUP["%d"%0],U_RF_GROUP["%d"%0]\
             = U_BB,V_BB,W,X,V_RF,U_RF
        for i in range(self.num_in):
            U_BB_GROUP["%d"%(i+1)],V_BB_GROUP["%d"%(i+1)],V_RF_GROUP["%d"%(i+1)],U_RF_GROUP["%d"%(i+1)]\
                ,W_GROUP["%d"%(i+1)],X_GROUP["%d"%(i+1)]= self.Inner_loop["%d"%i]\
                    (H,U_BB_GROUP["%d"%i],U_RF_GROUP["%d"%i],V_BB_GROUP["%d"%i],V_RF_GROUP["%d"%i],W_GROUP["%d"%i],X_GROUP["%d"%i],Y,rho)
        Y_new = self.Y_f(Y,X_GROUP["%d"%(self.num_in)],V_RF_GROUP["%d"%(self.num_in)],V_BB_GROUP["%d"%(self.num_in)],rho)
        return U_BB_GROUP["%d"%(self.num_in)],V_BB_GROUP["%d"%(self.num_in)],V_RF_GROUP["%d"%(self.num_in)],U_RF_GROUP["%d"%(self.num_in)]\
           ,W_GROUP["%d"%(self.num_in)],X_GROUP["%d"%(self.num_in)],Y_new

class Inner_layer(nn.Module):
    def __init__(self,K,d,M,N,N_RF,M_RF,P,sigma):
        super(Inner_layer,self).__init__()
        self.K = K
        self.d = d
        self.M = M 
        self.N = N 
        self.N_RF = N_RF
        self.M_RF = M_RF
        self.P = P
        self.sigma = sigma
        self.U_BB_f = U_BB_layer(self.K,self.M,self.M_RF,self.d)
        self.W_f = W_layer(self.K,self.d)
        self.V_RF_f = V_RF_layer(self.M, self.N,self.N_RF,self.K)
        self.U_RF_f = U_RF_layer(self.K,self.d,self.M,self.M_RF,self.N)
        self.V_BB_f = V_BB_layer(self.K,self.N_RF,self.N,self.d)
        self.X_f = X_layer(self.K,self.N,self.d,self.P)

    def forward(self,H,U_BB,U_RF,V_BB,V_RF,W,X,Y,rho):
        U_BB_new = torch.randn(self.K,2,self.M_RF,self.d)
        U_RF_new = torch.randn(self.K,2,self.M,self.M_RF)
        V_BB_new = torch.randn(self.K,2,self.N_RF,self.d)
        V_RF_new = torch.randn(2,self.N,self.N_RF)
        W_new = torch.randn(self.K,2,self.d,self.d)
        X_new = torch.randn(self.K,2,self.N,self.d)
        U_BB_new = self.U_BB_f(self.sigma,H,X,U_RF)
        W_new = self.W_f(U_BB_new,U_RF,H,X)
        V_RF_new = self.V_RF_f(H)
        U_RF_new = self.U_RF_f(H)
        V_BB_new = self.V_BB_f(V_RF_new,X,Y,8)
        X_new = self.X_f(H,U_RF_new,U_BB_new,W_new,V_RF_new,V_BB_new,Y,rho)
        return U_BB_new,V_BB_new,V_RF_new,U_RF_new,W_new,X_new

class U_BB_layer(nn.Module):
    def __init__(self,K,M,M_RF,d):
        super(U_BB_layer,self).__init__()
        self.K = K
        self.d = d
        self.M = M
        self.M_RF = M_RF

    def forward(self,sigma,H,X,U_RF):
        U_BB = torch.zeros(self.K,2,self.M_RF,self.d)
        A = torch.zeros(self.K,2,self.M,self.M)
        A = A_update(self.K,sigma,H,X,self.M)
        for k in range(self.K):
            temp = cmul(cmul(conjT(U_RF[k]),A[k]),U_RF[k])
            temp_pinv = cinv(temp)
            U_BB[k] = cmul(cmul(cmul(temp_pinv,conjT(U_RF[k])),H[k]),X[k])
        return U_BB
        
class W_layer(nn.Module):
    def __init__(self,K,d):
        super(W_layer,self).__init__()
        self.K = K
        self.d = d

    def forward(self,U_BB,U_RF,H,X):
        S = torch.zeros(self.K,2,self.d,self.d)
        W = torch.zeros(self.K,2,self.d,self.d)
        S = S_update(self.K,self.d,U_BB,U_RF,H,X)
        for k in range(self.K):
            W[k] = cinv(S[k])
        return W

class V_RF_layer(nn.Module):
    def __init__(self,M,N,N_RF,K):
        super(V_RF_layer,self).__init__()
        self.N = N
        self.N_RF = N_RF
        self.M = M
        self.K = K
        self.PP_NN = nn.Sequential(
            nn.Linear(2*self.K*self.M*self.N, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, self.N*self.N_RF),
            nn.Sigmoid()
        )

    def forward(self,H):
        V_RF_out = torch.zeros(2,self.N,self.N_RF)
        V_RF_temp = torch.zeros(2*self.N*self.N_RF)
        V_RF_result = torch.zeros(self.N*self.N_RF)
        H_temp = torch.zeros(2*self.K*self.M*self.N)
        for k in range(self.K):
            H_temp[k*2*self.M*self.N:k*2*self.M*self.N+self.M*self.N] = flat(H[k,0,:,:],1,self.M,self.N)
            H_temp[k*2*self.M*self.N+self.M*self.N:k*2*self.M*self.N+2*self.M*self.N] = flat(H[k,1,:,:],1,self.M,self.N)
        V_RF_result = self.PP_NN(H_temp)*3.1415926*2
        V_RF_out[0] = flat_inverse(torch.cos(V_RF_result),1,self.N,self.N_RF)
        V_RF_out[1] = flat_inverse(torch.sin(V_RF_result),1,self.N,self.N_RF)
        return V_RF_out

class U_RF_layer(nn.Module):
    def __init__(self,K,d,M,M_RF,N):
        super(U_RF_layer,self).__init__()
        self.K = K
        self.d = d
        self.M = M
        self.M_RF = M_RF
        self.N = N
        self.CP_NN = nn.Sequential(
            nn.Linear(2*self.K*self.M*self.N, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, self.K*self.M*self.M_RF),
            nn.Sigmoid()
        )

    def forward(self,H):
        U_RF_out = torch.zeros(self.K,2,self.M,self.M_RF)
        U_RF_temp = torch.zeros(2*self.K*self.M*self.M_RF)
        U_RF_result = torch.zeros(self.K*self.M*self.M_RF)
        H_temp = torch.zeros(2*self.K*self.M*self.N)
        for k in range(self.K):
            H_temp[k*2*self.M*self.N:k*2*self.M*self.N+self.M*self.N] = flat(H[k,0,:,:],1,self.M,self.N)
            H_temp[k*2*self.M*self.N+self.M*self.N:k*2*self.M*self.N+2*self.M*self.N] = flat(H[k,1,:,:],1,self.M,self.N)
        U_RF_result = self.CP_NN(H_temp)*3.1415926*2
        for k in range(self.K):
            U_RF_out[k,0,:,:] = flat_inverse(torch.cos(U_RF_result[k*self.M*self.M_RF:k*self.M*self.M_RF+self.M*self.M_RF]),1,self.M,self.M_RF)
            U_RF_out[k,1,:,:] = flat_inverse(torch.sin(U_RF_result[k*self.M*self.M_RF:k*self.M*self.M_RF+self.M*self.M_RF]),1,self.M,self.M_RF)
        return U_RF_out

class V_BB_layer(nn.Module):
    def __init__(self,K,N_RF,N,d):
        super(V_BB_layer,self).__init__()
        self.K = K
        self.N_RF = N_RF
        self.N = N
        self.d = d
        self.rho_V_BB_layer = nn.Parameter(torch.tensor(8.))
        self.params1 = {}
        self.params2 = {}
        self.params3 = {} 
        self.params4 = {}
        for i in range(self.K):
            self.params1["%d"%(i)] = nn.Parameter(torch.zeros(2,N,N))
            torch.nn.init.xavier_normal_(self.params1["%d"%(i)],gain = 0.01)
            self.params2["%d"%(i)] = nn.Parameter(torch.zeros(2,N_RF,N_RF))
            torch.nn.init.xavier_normal_(self.params2["%d"%(i)],gain = 0.01)
            self.params3["%d"%(i)] = nn.Parameter(torch.zeros(2,N,N_RF))
            torch.nn.init.xavier_normal_(self.params3["%d"%(i)],gain = 0.01)
            self.params4["%d"%(i)] = nn.Parameter(torch.zeros(2,N_RF,d))
            torch.nn.init.xavier_normal_(self.params4["%d"%(i)],gain = 0.01)
        self.params1 = nn.ParameterDict(self.params1)
        self.params2 = nn.ParameterDict(self.params2)
        self.params3 = nn.ParameterDict(self.params3)
        self.params4 = nn.ParameterDict(self.params4)

    def forward(self,V_RF,X,Y,rho):
        V_BB = torch.zeros(self.K,2,self.N_RF,self.d)
        # V_RF_temp = cmul(self.params1["%d"%(0)],cdiv(V_RF))+cmul(self.params2["%d"%(0)],V_RF)+self.params3["%d"%(0)]*0.001
        V_RF_temp = cpinv(V_RF)
        for k in range(self.K):
            V_BB[k] = cmul(V_RF_temp,(X[k]+self.rho_V_BB_layer * Y[k])) +self.params4["%d"%(k)]
        return V_BB

class X_layer(nn.Module):
    def __init__(self,K,N,d,P):
        super(X_layer,self).__init__()
        self.K = K
        self.N = N
        self.d = d
        self.P = P 
        self.rho_X_layer = nn.Parameter(torch.tensor(6.))
        self.mu = nn.Parameter(torch.tensor(1.))
        self.params1 = {}
        self.params2 = {}
        self.params3 = {}  
        self.params4 = {}  
        for i in range(1):
            self.params1["%d"%(i)] = nn.Parameter(torch.zeros(2,N,N))
            torch.nn.init.xavier_normal_(self.params1["%d"%(i)],gain = 0.001)
            self.params2["%d"%(i)] = nn.Parameter(torch.zeros(2,N,N))
            torch.nn.init.xavier_normal_(self.params2["%d"%(i)],gain = 0.001)
            self.params3["%d"%(i)] = nn.Parameter(torch.zeros(2,N,N))
            torch.nn.init.xavier_normal_(self.params3["%d"%(i)],gain = 0.001)
        for i in range(self.K):
            self.params4["%d"%(i)] = nn.Parameter(torch.zeros(2,N,d))
            torch.nn.init.xavier_normal_(self.params4["%d"%(i)],gain = 0.001)
        self.params1 = nn.ParameterDict(self.params1)
        self.params2 = nn.ParameterDict(self.params2)
        self.params3 = nn.ParameterDict(self.params3)
        self.params4 = nn.ParameterDict(self.params4)

    def forward(self,H,U_RF,U_BB,W,V_RF,V_BB,Y,rho):
        A_rho = torch.zeros(2,self.N,self.N)
        B_rho = torch.zeros(self.K,2,self.N,self.d)
        X = torch.ones(self.K,2,self.N,self.d)
        A_rho = A_rho_update(H,U_RF,U_BB,W,self.rho_X_layer,self.K,self.N)
        # A_rho_inv = torch.zeros(2,self.N,self.N)
        B_rho = B_rho_update(H,U_RF,U_BB,W,Y,V_RF,V_BB,self.rho_X_layer,self.N,self.K,self.d)
        I1 = torch.eye(self.N)
        I2 = torch.zeros(self.N,self.N)
        I = mcat(I1,I2)
        temp = A_rho + self.mu*I
        # A_rho_inv = cmul(self.params1["%d"%(0)],cdiv(temp)) + cmul(self.params2["%d"%(0)],temp) +self.params3["%d"%(0)]*0.01
        A_rho_inv = cinv(temp)
        for k in range(self.K):
            X[k] = cmul(A_rho_inv,B_rho[k]) + self.params4["%d"%(k)]
        return X

class Y_layer(nn.Module):
    def __init__(self, K, N, d):
        super(Y_layer, self).__init__()
        self.K = K
        self.N = N
        self.d = d
        self.rho_Y_layer = nn.Parameter(torch.tensor(8.))
        self.params1 = {}
        for i in range(self.K):
            self.params1["%d" % (i)] = nn.Parameter(torch.zeros(2, N, N))
            torch.nn.init.xavier_normal_(self.params1["%d" % (i)], gain = 0.1)
        self.params1 = nn.ParameterDict(self.params1)

    def forward(self, Y, X, V_RF, V_BB, rho):
        Y_out = torch.zeros(self.K, 2, self.N, self.d)
        for k in range(self.K):
            Y_out[k] = cmul(self.params1["%d"%(k)], Y[k]+1/self.rho_Y_layer*(X[k]-cmul(V_RF,V_BB[k])))
        return Y_out

def S_update(K,d,U_BB,U_RF,H,X):
    S = torch.zeros(K,2,d,d)
    I1 = torch.eye(d)
    I2 = torch.zeros(d,d)
    I = mcat(I1,I2)
    for k in range(K):
        S[k] = I - cmul(cmul(cmul(conjT(U_BB[k]),conjT(U_RF[k])),H[k]),X[k])
    return S

def A_update(K,sigma,H,X,M):
    A = torch.zeros(K,2,M,M)
    I1 = torch.eye(M)
    I2 = torch.zeros(M,M)
    I = mcat(I1,I2)
    for k in range(K):
        for j in range(K):
            A[k] += cmul(cmul(cmul(H[k],X[j]),conjT(X[j])),conjT(H[k]))
        A[k] += sigma**2*I
    return A

def flat(A,batch_size,Num_row,Num_rank):
    B = torch.zeros(Num_rank*Num_row)
    B = A.view(Num_rank*Num_row)
    return B

def flat_inverse(A,batch_size,Num_row,Num_rank):
    B = torch.zeros(Num_row,Num_rank)
    B = A.view(Num_row,Num_rank)
    return B

def A_rho_update(H,U_RF,U_BB,W,rho,K,N):
    A_rho = torch.zeros(2,N,N)
    I1 = torch.eye(N)
    I2 = torch.zeros(N,N)
    I = mcat(I1,I2)
    for j in range(K):
        temp = cmul(cmul(conjT(H[j]),U_RF[j]),U_BB[j])
        A_rho += cmul(cmul(temp,W[j]),conjT(temp))
    A_rho += 1/(2*rho) * I
    return A_rho

def B_rho_update(H,U_RF,U_BB,W,Y,V_RF,V_BB,rho,N,K,d):
    B_rho = torch.zeros(K,2,N,d)
    for k in range(K):
        B_rho[k] = cmul(cmul(cmul(conjT(H[k]),U_RF[k]),U_BB[k]),W[k])+0.5*(1./rho*cmul(V_RF,V_BB[k])-Y[k])
    return B_rho

def BCD_type(A, X, C, B, m, n, epsilon):
    X_temp = X
    Q = cmul(cmul(A, X_temp), C)
    termination = 1
    while (termination) > 0:
        for i in range(m):
            for j in range(n):
                b = cmul2(cmul2(mcat(A[0, i, i], A[1, i, i]), mcat(X_temp[0, i, j], X_temp[1, i, j])),
                          mcat(C[0, j, j], C[1, j, j])) - mcat(Q[0, i, j], Q[1, i, j]) + mcat(B[0, i, j], B[1, i, j])
                x = b / torch.sqrt(b[0] ** 2 + b[1] ** 2)
                Q = Q + cmul(cmul1((x - mcat(X_temp[0, i, j], X_temp[1, i, j])),
                                   mcat(A[0, :, i], A[1, :, i]).reshape((2, len(A[0, :, 0]), 1))),
                             mcat(C[0, j, :], C[1, j, :]).reshape((2, 1, len(C[0, 0, :]))))
                X_temp[0, i, j] = x[0]
                X_temp[1, i, j] = x[1]
        termination -= 1
    return X_temp

def BCD_type_B(X, Y, V_BB, rho, K, N, N_RF):
    B = torch.zeros((2, N, N_RF))
    for k in range(K):
        B += cmul((X[k] + rho * Y[k]), conjT(V_BB[k]))
    return B

def BCD_type_C(V_BB, K, N_RF):
    C = torch.zeros(2, N_RF, N_RF)
    for k in range(K):
        C += cmul(V_BB[k], conjT(V_BB[k]))
    return C