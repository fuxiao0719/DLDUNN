from lib.config import cfg, args
import numpy as np
import random
from numpy.linalg import eigh

def update_upsilon(U_RF, H, X, sigma, M, M_RF, K):
    upsilon_temp = np.zeros((K,M_RF,M_RF),complex)
    for k in range(K):
        upsilon_item = np.zeros((M,M),complex)
        for j in range(K):
            if j==k:
                continue
            else:
                upsilon_item += H[k].dot(X[j]).dot(X[j].conjugate().T).dot(H[k].conjugate().T)
        upsilon_item += (sigma**2) * (np.eye(M)+0j*np.eye(M))
        upsilon_temp[k] = (U_RF[k].conjugate().T).dot(upsilon_item).dot(U_RF[k])
    return upsilon_temp

def update_upsilon_target(U_RF, H, V_RF, V_BB, sigma, M, M_RF, K):
    upsilon_temp = np.zeros((K,M_RF,M_RF),complex)
    for k in range(K):
        upsilon_item = np.zeros((M,M),complex)
        for j in range(K):
            if j==k:
                continue
            else:
                upsilon_item += H[k].dot(V_RF).dot(V_BB[j]).dot(V_BB[j].conjugate().T).dot(V_RF.conjugate().T).dot(H[k].conjugate().T)
        upsilon_item += (sigma**2) * (np.eye(M)+0j*np.eye(M))
        upsilon_temp[k] = (U_RF[k].conjugate().T).dot(upsilon_item).dot(U_RF[k])
    return upsilon_temp

def Ak(H, X, sigma, K, M, k):
    Ak = np.zeros((M,M),complex)
    for j in range(K):
        Ak += H[k].dot(X[j]).dot(X[j].conjugate().T).dot(H[k].conjugate().T)
    Ak = Ak + sigma**2*(np.eye(M)+0j*np.eye(M))
    return Ak

def update_A_rho(H, U_RF, U_BB, W, rho, K, N):
    A_rho = np.zeros((N,N),complex)
    for j in range(K):
        A_rho += (H[j].conjugate().T).dot(U_RF[j]).dot(U_BB[j]).dot(W[j]).dot(U_BB[j].conjugate().T).dot(U_RF[j].conjugate().T).dot(H[j])
    A_rho += 1/(2*rho)* (np.eye(N)+0j*np.eye(N))
    return A_rho

def update_B_rho(H, U_RF, U_BB, W, Y, V_RF, V_BB, rho, K, N, d):
    B_rho = np.zeros((K,N,d),complex)
    for k in range(K):
        B_rho[k] = (H[k].conjugate().T).dot(U_RF[k]).dot(U_BB[k]).dot(W[k])+0.5*(1/rho*V_RF.dot(V_BB[k])-Y[k])
    return B_rho

def BCD_type_B(X, Y, V_BB, rho, K, N, N_RF):
    B = np.zeros((N,N_RF),complex)
    for k in range(K):
        B += (X[k]+rho*Y[k]).dot(V_BB[k].conjugate().T)
    return B

def BCD_type_C(V_BB, K, N_RF):
    C = np.zeros((N_RF,N_RF),complex)
    for k in range(K):
        C += V_BB[k].dot(V_BB[k].conjugate().T)
    return C

def BCD_type(A, X, C, B, m, n, epsilon):
    X_temp = X
    Q = A.dot(X_temp).dot(C)
    phi_new = np.trace((X_temp.conjugate().T.dot(A).dot(X_temp).dot(C)))-2*np.real(np.trace((X_temp.conjugate().T.dot(B))))
    termination = 10
    while(termination)>epsilon:
        for i in range(m):
            for j in range(n):
                b = A[i,i]*X_temp[i,j]*C[j,j]-Q[i,j]+B[i,j]
                x = b / abs(b)
                Q = Q + (x - X_temp[i,j])*((A[:,i].reshape((len(A[:,0]),1))).dot(C[j,:].reshape((1,len(C[0,:])))))
                X_temp[i,j] = x
        phi_old = phi_new
        phi_new = np.trace((X_temp.conjugate().T).dot(A).dot(X_temp).dot(C))-2*np.real(np.trace(X_temp.conjugate().T.dot(B)))
        termination = abs(phi_new-phi_old)/abs(phi_old)
    return X_temp

def Bisection_mu(B_rho, A_rho, P, K, N, epsilon):
    mu_min = 0
    mu_max = 100000
    mu = mu_min 
    vals, vecs = eigh(A_rho)
    a = np.diag(vals)
    U_rho = vecs
    b = np.zeros((K,N),complex)
    temp = 0
    for k in range(K):
        for i in range(N):
            b[k,i] = U_rho.conjugate().T.dot(B_rho[k]).dot(B_rho[k].conjugate().T).dot(U_rho)[i,i]
            temp = temp + b[k,i]/(a[i,i]+mu)**2  
    if temp<=P:
        return mu
    else:
        while(abs(temp-P)>epsilon*(1e-3)): 
            mu = (mu_min + mu_max) / 2
            temp = 0
            for k in range(K):
                for i in range(N):
                    temp = temp + b[k,i]/(a[i,i]+mu)**2
            if(abs(temp)>P):
                mu_min = mu
            else:
                mu_max = mu
        return mu

def Augmented_lagrange(U_BB, U_RF, H, W, X, V_BB, V_RF, upsilon, rho, d, K,Y):
    L = 0
    for k in range(K):
        Ek_item = np.eye(d)+0j*np.eye(d) - (U_BB[k].conjugate().T).dot(U_RF[k].conjugate().T).dot(H[k]).dot(X[k])
        Ek = Ek_item.dot(Ek_item.conjugate().T)+(U_BB[k].conjugate().T).dot(upsilon[k]).dot(U_BB[k])
        L = L + (np.log2(np.linalg.det(W[k]))-np.trace(W[k].dot(Ek))+d)
        L_item = X[k]-V_RF.dot(V_BB[k])+rho*Y[k]
        L = L - 1/(2*rho)*np.trace(L_item.dot(L_item.conjugate().T))
    return L

def target(U_RF, H, V_RF, V_BB, d, upsilon, P, K):
    target = 0
    for k in range(K):
        target_item = np.eye(M_RF)+0j*np.eye(M_RF)+(U_RF[k].conjugate().T).dot(H[k]).dot(V_RF).dot(V_BB[k]).dot(V_BB[k].conjugate().T).dot(V_RF.conjugate().T).dot(H[k].conjugate().T).dot(U_RF[k]).dot(np.linalg.inv(upsilon[k]))
        target += np.log(np.linalg.det(target_item))
    return target

if __name__=="__main__":
    K = cfg.K
    d = cfg.d
    M = cfg.M
    N = cfg.N
    M_RF = cfg.M_RF
    N_RF = cfg.N_RF
    print('K:{0},d:{1},M:{2},N:{3},M_RF:{4},N_RF:{5}'.format(K,d,M,N,M_RF,N_RF))
    c = 0.8
    sigma = cfg.sigma
    P = pow(10,2)
    max_iteration = cfg.max_iteration
    number = 0
    spectral_efficiency = []
    average_number = cfg.average_number
    while number < average_number:
        rho = 100/N
        eta = 1e-3
        epsilon = 1e-5
        # initialize Matrix
        V_RF = np.random.randn(N,N_RF)
        V_RF = np.sin(V_RF)+1j*np.cos(V_RF)
        V_BB = np.zeros((K,N_RF,d),complex)
        U_RF = np.random.randn(K,M,M_RF)
        U_RF = np.sin(U_RF)+1j*np.cos(U_RF)
        U_BB = np.random.randn(K,M_RF,d)+1j*np.random.randn(K,M_RF,d)
        X = np.random.randn(K,N,d)+1j*np.random.randn(K,N,d)
        Y = 10*np.random.randn(K,N,d)+10j*np.random.randn(K,N,d)
        W = np.random.randn(K,d,d)+1j*np.random.randn(K,d,d)

        # ComplexGuass channel
        H = np.random.randn(K,M,N)+1j*np.random.randn(K,M,N)

        for k in range(K):
            V_BB[k] = np.sqrt(1/2)*(np.random.randn(N_RF,d)+1j*np.random.randn(N_RF,d))
            X[k] = V_RF.dot(V_BB[k])
            Y[k] = (np.random.randn(N,d)+1j*np.random.randn(N,d))

        sum_item = np.sum([np.trace(np.dot(X[i],X[i].conjugate().T)) for i in range(K)])
        for k in range(K):
            V_BB[k] = V_BB[k] *np.sqrt(P/sum_item)
            X[k] = V_RF.dot(V_BB[k])

        # initialize Matrix that loop needs
        upsilon = np.zeros((K,M_RF,M_RF),complex)
        A = np.zeros((K,M,M),complex)
        A_rho = np.zeros((N,N),complex)
        B_rho = np.zeros((N,d),complex)
        upsilon = update_upsilon(U_RF, H, X, sigma, M, M_RF, K)

        #PDD method
        h_x = 10
        L_old = 1   
        L_new = 10
        while h_x > 1e-4:
            iteration = 0 
            while((abs(L_new-L_old)/abs(L_old))>epsilon and iteration < max_iteration):
                for k in range(K):
                    U_BB[k] =  np.linalg.pinv((U_RF[k].conjugate().T).dot(Ak(H, X, sigma, K, M, k)).dot(U_RF[k])).dot(((U_RF[k].conjugate().T).dot(H[k]).dot(X[k])))
                for k in range(K):
                    W[k] = np.linalg.inv(np.eye(d)+0j*np.eye(d)-(U_BB[k].conjugate().T).dot((U_RF[k].conjugate().T.dot(H[k]).dot(X[k]))))
                V_RF = BCD_type((np.eye(N)+0j*np.eye(N)), V_RF, BCD_type_C(V_BB, K, N_RF), BCD_type_B(X, Y, V_BB, rho, K, N, N_RF), N, N_RF, epsilon)
                for k in range(K):
                    U_RF[k] = BCD_type(Ak(H, X, sigma, K, M, k), U_RF[k], U_BB[k].dot(W[k]).dot(U_BB[k].conjugate().T), H[k].dot(X[k]).dot(W[k]).dot(U_BB[k].conjugate().T), M, M_RF, epsilon)
                for k in range(K):
                    V_BB[k] = np.linalg.pinv(V_RF).dot(X[k]+rho*Y[k])
                B_rho = update_B_rho(H, U_RF, U_BB, W, Y, V_RF, V_BB, rho, K, N, d)
                A_rho = update_A_rho(H, U_RF, U_BB, W, rho, K, N)
                for k in range(K):
                    mu = Bisection_mu(B_rho, A_rho, P, K, N, epsilon)
                    X[k] = np.linalg.inv(A_rho+mu*(np.eye(N)+0j*np.eye(N))).dot(B_rho[k])
                upsilon = update_upsilon(U_RF, H, X , sigma, M, M_RF, K)
                L_old = L_new
                L_new = Augmented_lagrange(U_BB, U_RF, H, W, X, V_BB, V_RF, upsilon, rho, d, K,Y)
                iteration += 1
            h_x = np.max([np.linalg.norm(X[i]-V_RF.dot(V_BB[i]),ord=np.inf) for i in range(K)]) 
            if h_x<= eta:
                for k in range(K):
                    Y[k] = Y[k] + 1/rho*(X[k]-V_RF.dot(V_BB[k]))
            else:
                rho = c * rho
            eta = 0.9 * h_x
            epsilon = c * epsilon
        number += 1
        spectral_efficiency.append(target(U_RF, H, V_RF, V_BB, d, update_upsilon_target(U_RF, H, V_RF, V_BB, sigma, M, M_RF, K), P, K))
        if number%5==0:
            print('{}/{} is done'.format(number,average_number))
    print('spectral efficiency:{}'.format(np.sum(spectral_efficiency)/len(spectral_efficiency)))
    print('Done')