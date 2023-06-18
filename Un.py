
from cmath import exp, sqrt, cos, sin
import numpy as np

class Dif_eq:
    rtol = 1e-6
    atol = 1e-12
    method_order =3
    B=[1/6, 4/6, 1/6]
    C=[0,1/2,1]
    A= [[0,0,0],[1/2,0,0],[-1,2,0]]
    def __init__(self, m, p, k, y_0, x_0, x_1, h=None):
        self.m=m
        self.p=p
        self.k=k
        self.y0=np.array(y_0)
        self.x0=x_0
        self.x1=x_1
        self.h=h
        self.lambda1, self.lambda2, self.C1, self.C2 = self.analitic()


    def change(self, h=None,m=None, p=None, k=None, x1=None, x0=None, y0=None):
        an_change=0
        if(h is not None):
            self.h=h
        if(m is not None):
            self.m=m
            an_change=1
        if(p is not None):
            self.p=p
            an_change=1
        if(k is not None):
            self.k=k
            an_change=1
        if an_change:
            self.lambda1, self.lambda2, self.C1, self.C2 = self.analitic()
        if(x1 is not None):
            self.x1=x1
        if(x0 is not None) and (y0 is None):
            return('error')
        elif(x0 is not None):
            self.x0=x0
            self.y0=y0
        elif (y0 is not None):
            self.y0=y0

    # Правая часть системы ДУ
    def f_mean(self, x: float, y: list):
        return [
        y[1], 
        -1/self.m*(self.p*y[1]+self.k*y[0])
        ]
    
    def K_find(self, x: float, h: float, y: float): #поиск K
        K=[]
        for i in range(0, self.method_order):
            Y=np.array([0, 0])
            for j in range(0,i):
                Y=Y+self.A[i][j]*np.array(K[j])
            K.append(self.f_mean(x+self.C[i]*h, y+h*Y))
        return(K)
    
    def y_find(self, K, h: float, y: float, B_=B): #поиск соответствующих значений y[0], y[1]
        y1=0
        for i in range(0, self.method_order):
            y1+=np.array(K[i])*B_[i]
        y1=y+h*np.array(y1)
        return(y1)

    def methodRK_fixstep(self): #ищет значение в x1
        if self.h is None:
            return('Error')
        y_means = [self.y0]
        x_mean=[self.x0]
        while x_mean[-1]+self.h<=self.x1:
            K=self.K_find(x_mean[-1], self.h, y_means[-1])
            y_means.append(self.y_find(K, self.h, y_means[-1]))
            x_mean.append(x_mean[-1]+self.h)
        if(x_mean[-1]<self.x1 and x_mean[-1]+self.h>self.x1):
            K=self.K_find(x_mean[-1], self.x1-x_mean[-1], y_means[-1])
            y_means.append(self.y_find(K, self.x1-x_mean[-1], y_means[-1]))
            x_mean.append(self.x1)
        return(x_mean, y_means)
    
    def analitic(self):
        D=self.p*self.p-4*self.k*self.m
        lambda1=(sqrt(D)-self.p)/(2*self.m)
        lambda2=-(sqrt(D)+self.p)/(2*self.m)
        C1=(lambda2*self.y0[0]-self.y0[1])/((lambda2-lambda1)*exp(lambda1*self.x0))
        C2=(self.y0[1]-lambda1*self.y0[0])/((lambda2-lambda1)*exp(lambda2*self.x0))
        return(lambda1, lambda2, C1, C2)
    
    def y_true(self, x):
        if self.D:
            return(self.C1*exp(self.lambda1*x)+self.C2*exp(self.lambda2*x))
        return(exp(self.lambda1*x)*(self.C1*cos(self.lambda2*x)+self.C2*sin(self.lambda2*x)))
    
    def B_dense(self, i):
        return([i- 5/6 * i**2, 4/6*i**2, 1/6 * i**2])

    
    def methodRK_forsomex(self, X):
        if self.h is None:
            return('Error. No h')
        if isinstance(X, int) or isinstance(X, float):
            x_rem=self.x1
            self.change(x1=X)
            res = self.methodRK_fixstep()
            self.change(x1=x_rem)
            return(res)
        X=sorted(X)
        if(X[0]<self.x0):
            return('Error. x out of range')
        x_rem=self.x1
        if(X[-1]>self.x1):
            self.change(x1=X[-1])
        Y=[]
        y_means = [self.y0]
        x_mean=[self.x0]
        while x_mean[-1]+self.h<=self.x1:
            K=self.K_find(x_mean[-1], self.h, y_means[-1])
            y_means.append(self.y_find(K, self.h, y_means[-1]))
            x_mean.append(x_mean[-1]+self.h)
            while(len(X) and X[0]<=x_mean[-1]):
                x_find = X.pop(0)
                B_new = self.B_dense((x_find-x_mean[-2])/self.h)
                Y.append(self.y_find(K, self.h, y_means[-2], B_new)[0])

        if(x_mean[-1]<self.x1 and x_mean[-1]+self.h>self.x1):
            K=self.K_find(x_mean[-1], self.x1-x_mean[-1], y_means[-1])
            y_means.append(self.y_find(K, self.x1-x_mean[-1], y_means[-1]))
            x_mean.append(self.x1)
            while(len(X) and X[0]<=x_mean[-1]):
                x_find = X.pop(0)
                B_new = self.B_dense((x_find-x_mean[-2])/self.h)
                Y.append(self.y_find(K, self.h, y_means[-2], B_new)[0])
        return(x_mean, y_means, Y)
    
    # Поиск начального шага
    def firstStep(self):
        K1 = self.f_mean(self.x0, self.y0)
        delta = (1/self.x1) ** (self.method_order+1) + np.linalg.norm(K1) ** (self.method_order+1)
        h = (self.rtol / delta) ** (1 / (self.method_order+1))
        if (K1.count(0) <= 2):
            return h
        u1 = [self.y0[i] + h * K1[i] for i in range(len(K1))]
        K2 = self.f_mean(self.x0+h, u1)
        delta = (1/max(abs(self.x0+h), abs(self.x0+1/2**4))) ** (self.method_order+1) + np.linalg.norm(K2) ** (self.method_order+1)
        return min(h, (self.rtol / delta) ** (1 / (self.method_order+1)))
        
    def localErrorr(self, xk, yk, h): # Локальная погрешность Рунге
        y_1 = self.y_find(self.K_find(xk, h, yk), h, yk)
        y_2_1 = self.y_find(self.K_find(xk, h/2, yk), h/2, yk)
        y_2 = self.y_find(self.K_find(xk+h/2, h/2, y_2_1), h/2, y_2_1)
        return (np.linalg.norm(
            ([(1/(1-2**(-self.method_order)))* abs(y_2[i] - y_1[i]) for i in range(len(y_2))])), y_1, y_2, y_2_1
        )  
    
    def h_choise (self, r, y_1, y_2, y_2_1, h, hmax): #Алгоритм удвоения и деления шага пополам
        k = max(np.linalg.norm(np.array(y_1)), np.linalg.norm(np.array(y_2))) * self.rtol + self.atol
        if r/h > k * 2**self.method_order:
            return (False, h/2, y_2_1)
        if k < r/h:
            return (True, h/2, y_2)
        if k / (2**(self.method_order+1)) <= r/h:
            return (True, h, y_1)
        return (True, min(h * 2, hmax), y_1)    

    # Автоматический выбор шага
    def Auto_h_method(self, h_max):
        h = [self.firstStep()]
        x=[self.x0]
        y=[self.y0]
        while x[-1] < self.x1:
            r, y_1, y_2, y_2_1 = self.localErrorr(x[-1], y[-1], h[-1])
            ch, new_h, new_y = self.h_choise(r, y_1, y_2, y_2_1, h[-1], h_max)
            if not ch:
                h[-1] = new_h
                y[-1] = new_y
            else:
                if x[-1]+h[-1] >= self.x1:
                    K=self.K_find(x[-1], self.x1 - x[-1], y[-1])
                    y.append(self.y_find(K, self.x1 - x[-1], y[-1]))
                    x.append(self.x1)
                    break
                x.append(x[-1]+h[-1]) 
                h.append(new_h)
                y.append(new_y)
        return(x, y, h)      
        