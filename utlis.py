import torch
from model import * 
from complex_matrix import *
import os

USE_GPU = False
dtype = torch.float32 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if USE_GPU == False:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
print('using device:', device)
print_every = 1

def update_upsilon(U_RF, H,V_RF, V_BB, sigma, M, M_RF, K):
    upsilon_temp = torch.zeros((K,2,M_RF,M_RF))
    for k in range(K):
        upsilon_item = torch.zeros((2,M,M))
        for j in range(K):
            if j==k:
                continue
            else:
                upsilon_item += cmul(cmul(cmul(cmul(cmul(H[k],V_RF),V_BB[j]),conjT(V_BB[j])),conjT(V_RF)),conjT(H[k]))
        upsilon_item += (sigma**2)* mcat(torch.eye(M),torch.zeros((M,M)))
        upsilon_temp[k] = cmul(cmul(conjT(U_RF[k]),(upsilon_item)),(U_RF[k]))
    return upsilon_temp

def produce_data(K, M, N, batch_size):
    H_set = torch.zeros((batch_size,K,2,M,N))
    H = torch.zeros((K,2,M,N))
    for i in range(batch_size):
        for k in range(K):
            H_temp = np.random.randn(M,N)+1j*np.random.randn(M,N)
            H[k] = c2m(H_temp)
        H = H.to(dtype=dtype,device=device)
        H_set[i] = H[:,:,:,:]
    return H_set

def loss_f(H,V_BB,V_RF,U_RF, K, M_RF, M, sigma):
    loss = 0
    I1 = torch.eye(M_RF)
    I2 = torch.zeros((M_RF,M_RF))
    I = mcat(I1,I2)
    for k in range(K):
        loss_item = I + cmul(cmul(cmul(cmul(cmul(cmul(cmul(cmul(conjT(U_RF[k,:,:,:]),(H[k,:,:,:])),V_RF[:,:,:]),V_BB[k,:,:,:]),conjT(V_BB[k,:,:,:])),conjT(V_RF[:,:,:])),conjT(H[k,:,:,:])),U_RF[k,:,:,:]),cinv(update_upsilon(U_RF, H,V_RF, V_BB, sigma, M, M_RF, K)[k,:,:,:]))
        loss -= torch.log(cdet(loss_item))
    return loss

def train(model,optimizer,U_BB0,U_RF0,V_BB0,V_RF0,W0,X0,Y0,K,M,N,M_RF,sigma,epochs):
    model = model.to(device=device,dtype=dtype) 
    val_batch_size = 64
    train_batch_size = 640
    H_val = produce_data(K,M,N,val_batch_size)
    i = 0
    loss = 0
    for e in range(epochs):
        print("epochs %d"%e) 
        H_train = produce_data(K,M,N,train_batch_size)
        for j in range(train_batch_size):
            model.train() 
            U_BB,U_RF,V_BB,V_RF = model(H_train[j],U_BB0,U_RF0,V_BB0,V_RF0,W0,X0,Y0)
            loss += loss_f(H_train[j],V_BB,V_RF,U_RF,K,M_RF,M,sigma)/8
            if i%8==0: 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = 0
            if i%64==0:
                result = 0
                for t in range(64):
                    U_BB,U_RF,V_BB,V_RF = model(H_val[t],U_BB0,U_RF0,V_BB0,V_RF0,W0,X0,Y0)
                    result += loss_f(H_val[t],V_BB,V_RF,U_RF,K,M_RF,M,sigma)/64
                print(result)
            i += 1