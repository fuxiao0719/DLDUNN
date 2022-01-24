import torch
from lib.config import cfg, args
from lib.model import * 
from lib.complex_matrix import *
from lib.utlis import *

if __name__ == "__main__":
    K = cfg.K
    d = cfg.d
    M = cfg.M
    N = cfg.N
    M_RF = cfg.M_RF
    N_RF = cfg.N_RF
    print('K:{0},d:{1},M:{2},N:{3},M_RF:{4},N_RF:{5}'.format(K,d,M,N,M_RF,N_RF))
    P = cfg.power
    sigma = cfg.sigma
    D_in = cfg.D_in
    D_out = cfg.D_out
    learning_rate = 0.001
    temp = torch.randn((N,N_RF))
    V_RF = torch.zeros(2,N,N_RF)
    V_RF[0] = torch.cos(temp)
    V_RF[1] = torch.sin(temp)
    V_RF = V_RF.to(dtype=dtype,device=device)
    V_BB = torch.randn((K,2,N_RF,d))
    temp = torch.randn((K,M,M_RF))
    U_RF = torch.zeros(K,2,M,M_RF)
    U_RF[:,0,:,:] = torch.cos(temp)
    U_RF[:,1,:,:] = torch.sin(temp)
    U_RF = U_RF.to(dtype=dtype,device=device)
    V_BB = V_BB.to(dtype=dtype,device=device)
    sum_temp = 0
    for k in range(K):
        sum_temp += torch.trace(cmul(conjT(cmul(V_RF,V_BB[k])),cmul(V_RF,V_BB[k]))[0])
    V_BB = V_BB * torch.sqrt(P/sum_temp)
    W = torch.randn(K,2,d,d)
    W= W.to(dtype=dtype,device=device)
    X = torch.randn(K,2,N,d)
    for k in range(K):
        X[k] = cmul(V_RF,V_BB[k])
    U_BB = torch.randn(K,2,N_RF,d)
    U_BB = U_BB.to(dtype=dtype,device=device)
    Y = torch.randn(K,2,N,d)
    DLDUNN = model(K,d,M,N,N_RF,M_RF,P,sigma,D_in,D_out,device)
    optimizer = torch.optim.Adam(DLDUNN.parameters(), lr=learning_rate)
    train(DLDUNN, optimizer,U_BB,U_RF,V_BB,V_RF,W,X,Y,K,M,N,M_RF,sigma,epochs=10000)