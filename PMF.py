import numpy as np
import matplotlib.pyplot as plt
from dataprocess import z,total_u,total_p

def PMF(data=z, factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, regU = 0.01 ,regI = 0.01 ,plot=False):
    P = np.random.rand(total_u, factors) / 3
    Q = np.random.rand(total_p, factors) / 3
    y = []
    iteration = 0
    last_loss = 100
    while iteration < maxIter:
        loss = 0
        for i in range(data.shape[0]):
            u, p, s = data[i]
            error = s - np.dot(P[u], Q[p])
            loss += error ** 2/50
            pp = P[u]
            qq = Q[p]
            P[u] += LRate *  (error * qq - regU*pp)
            Q[p] += LRate * (error * pp - regI * qq)
        loss += regU*(P*P).sum() +regI*(Q*Q).sum()
        iteration += 1
        y.append(loss)
        delta_loss = last_loss - loss
        print('iter = {}, loss = {}, delta_loss = {}, LR = {}'.format(iteration, loss, delta_loss, LRate))
        if abs(last_loss) > abs(loss):
            LRate *= 1.05
        else:
            LRate *= 0.5

        if abs(delta_loss) < abs(GD_end):
            print('the diff in loss is {}, so the GD stops'.format(delta_loss))
            break
        last_loss = loss
    if plot:
        plt.plot(y)
        plt.show()
    return P.dot(Q.T)

# MF_train = PMF(data=train, maxIter=100,plot=True)
# caculate MSE, in sample and out sample

# caculate_mse(PMF(data=z, factors=30, maxIter=100, LRate=0.03, GD_end=1e-3, regU = 0.01 ,regI = 0.01 ))
# draw_mse(PMF,50)
# print('cal')
# a = PMF()
# from cm import drawcm
# drawcm(a,test,'PMF')
#
# def rec(data,uid,n,rawId= False):
#     if uid in range(total_u):
#         s = data[uid]
#         top_N = np.argpartition(data[uid],-n)[-n:]
#         print('the top{} recommanded products for user {} is {}'.format(n,uid,top_N))
#         if rawId == True:
#             print('the real ID is {}'.format(pid2PID[top_N]))
#     else:
#         print('this user has not bought anything, plz use other methods')
#     return top_N
# # try one
# c = rec(a,9,10,True)
# print(c)