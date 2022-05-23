# from tkinter import N
import numpy as np
from numpy import sqrt
import time

MAX_ITER = 100

class Vanila():
    def __init__(self, N, step_size, cost_func):
        self.cost_func = cost_func
        self.N = N
        self.step_size = step_size

    def run(self):
        mean = np.zeros(self.N)
        C = np.diag(np.ones(self.N))

        lam = 100#4 + int(np.log(self.N) * 3)
        mu = 10#int(lam/2)
        best = 100

        start = time.time()
        for num in range(MAX_ITER):
            z_sample = np.random.multivariate_normal(mean, C, lam)
            z_sel = np.zeros(self.N)
            
            sample = mean + self.step_size * z_sample

            score = [self.cost_func(s) for s in sample]

            selected = [[j,score[j]] for i, j in enumerate(np.argsort(score)) if i<mu]

            for s in selected:
                if s[1]<best:
                    best = s[1]
                    print(f"#{num} f({sample[s[0]]}) = {s[1]}")
                    print(f"{time.time() - start} s")
                z_sel = z_sel + (1/mu) * z_sample[s[0]]
            mean = mean + self.step_size * z_sel
                
        print("DONE")

class RankMu():
    def __init__(self, N, step_size, cost_func):
        self.cost_func = cost_func
        self.N = N
        self.step_size = step_size

    def run(self):
        mean = np.zeros(self.N)
        C = np.diag(np.ones(self.N))

        lam = 100
        mu = 10
        best = 100

        w = [1/mu for _ in range(mu)]
        mu_eff = 1 / sum([x*x for x in w])
        
        cmu = min(1, mu_eff/(self.N**2))

        start = time.time()
        for num in range(MAX_ITER):
            y_sample = np.random.multivariate_normal(mean, C, lam)
            y_sel = np.zeros(self.N)
            y_square = np.zeros((self.N,self.N))
            
            sample = mean + self.step_size * y_sample

            score = [self.cost_func(s) for s in sample]

            selected = [[j,score[j]] for i, j in enumerate(np.argsort(score)) if i<mu]

            for s in selected:
                if s[1]<best:
                    best = s[1]
                    print(f"#{num} f({sample[s[0]]}) = {s[1]}")
                    print(f"{time.time() - start} s")
                y_sel = y_sel + (1/mu) * y_sample[s[0]]
                y_square = y_square + (1/mu) * y_sample[s[0]].reshape(self.N, 1) * y_sample[s[0]].reshape(1, self.N)
            mean = mean + self.step_size * y_sel

            # C = (1 - c_cov) * C + c_cov * (1 / mu_eff) * z_sel.reshape(self.N,1) * z_sel.reshape(1,self.N)
            C = (1 - cmu) * C + cmu * y_square
                    
        print("DONE")

class RankOne():
    def __init__(self, N, step_size, cost_func):
        self.cost_func = cost_func
        self.N = N
        self.step_size = step_size

    def run(self):
        mean = np.zeros(self.N)
        C = np.diag(np.ones(self.N))

        lam = 100#4 + int(np.log(self.N) * 3)
        mu = 10#int(lam/2)
        best = 100

        w = [1/mu for _ in range(mu)]
        mu_eff = 1 / sum([x*x for x in w])
        c1 = 2 / self.N**2

        start = time.time()
        for num in range(MAX_ITER):
            y_sample = np.random.multivariate_normal(mean, C, lam)
            y_sel = np.zeros(self.N)
            
            sample = mean + self.step_size * y_sample

            score = [self.cost_func(s) for s in sample]

            selected = [[j,score[j]] for i, j in enumerate(np.argsort(score)) if i<mu]

            for s in selected:
                if s[1]<best:
                    best = s[1]
                    print(f"#{num} f({sample[s[0]]}) = {s[1]}")
                    print(f"{time.time() - start} s")
                y_sel = y_sel + (1/mu) * y_sample[s[0]]
            x = mean + self.step_size * y_sel
            y_g1 = (x - mean) / self.step_size
            mean = x
            # C = (1 - c_cov) * C + c_cov * (1 / mu_eff) * z_sel.reshape(self.N,1) * z_sel.reshape(1,self.N)
            C = (1 - c1) * C + c1 * y_g1.reshape(self.N, 1) * y_g1.reshape(1, self.N)
                    
        print("DONE")

class RankOne():
    def __init__(self, N, step_size, cost_func):
        self.cost_func = cost_func
        self.N = N
        self.step_size = step_size

    def run(self):
        mean = np.zeros(self.N)
        C = np.diag(np.ones(self.N))

        lam = 100#4 + int(np.log(self.N) * 3)
        mu = 10#int(lam/2)
        best = 100

        w = [1/mu for _ in range(mu)]
        mu_eff = 1 / sum([x*x for x in w])
        c1 = 2 / self.N**2

        p_c = np.zeros((self.N, 1))

        start = time.time()
        for num in range(MAX_ITER):
            y_sample = np.random.multivariate_normal(mean, C, lam)
            y_sel = np.zeros(self.N)
            
            sample = mean + self.step_size * y_sample

            score = [self.cost_func(s) for s in sample]

            selected = [[j,score[j]] for i, j in enumerate(np.argsort(score)) if i<mu]

            for s in selected:
                if s[1]<best:
                    best = s[1]
                    print(f"#{num} f({sample[s[0]]}) = {s[1]}")
                    print(f"{time.time() - start} s")
                y_sel = y_sel + (1/mu) * y_sample[s[0]]
            x = mean + self.step_size * y_sel
            y_g1 = (x - mean) / self.step_size
            
            p_c = (1 - cc)
            mean = x
            
            C = (1 - c1) * C + c1 * y_g1.reshape(self.N, 1) * y_g1.reshape(1, self.N)
                    
        print("DONE")

class RankOne_Cum_RankMu():
    def __init__(self, N, step_size, cost_func):
        self.cost_func = cost_func
        self.N = N
        self.step_size = step_size

    def run(self):
        mean = np.zeros(self.N)
        C = np.diag(np.ones(self.N))

        lam = 100#4 + int(np.log(self.N) * 3)
        mu = 10#int(lam/2)
        best = 100

        p_c = np.zeros(self.N)

        w = [1/mu for _ in range(mu)]
        mu_eff = 1 / sum([x*x for x in w])

        c_c = 4/self.N
        c_cov = mu_eff/(self.N**2)
        mu_cov = mu_eff

        start = time.time()
        for num in range(MAX_ITER):
            z_sample = np.random.multivariate_normal(mean, C, lam)
            z_sel = np.zeros(self.N)
            Z = np.zeros(self.N)
            
            sample = mean + self.step_size * z_sample

            score = [self.cost_func(s) for s in sample]

            selected = [[j,score[j]] for i, j in enumerate(np.argsort(score)) if i<mu]

            for s in selected:
                if s[1]<best:
                    best = s[1]
                    print(f"#{num} f({sample[s[0]]}) = {s[1]}")
                    print(f"{time.time() - start} s")
                z_sel = z_sel + (1/mu) * z_sample[s[0]]
                Z = Z + (1/mu) * z_sample[s[0]].reshape(self.N,1) *  z_sample[s[0]].reshape(1,self.N) 
            mean = mean + self.step_size * z_sel

            p_c = (1 - c_c) * p_c + sqrt(1 - (1 - c_c)**2) * sqrt(mu_eff) * z_sel
            C = (1 - c_cov) * C + c_cov/mu_cov * p_c.reshape(self.N, 1) * p_c.reshape(1, self.N) + c_cov * (1 - 1/mu_cov) * Z
                    
        print("DONE")

class CMAES():
    def __init__(self, N, step_size, cost_func):
        self.cost_func = cost_func
        self.N = N
        self.step_size = step_size

    def run(self):
        mean = np.zeros(self.N)
        C = np.diag(np.ones(self.N))

        lam = 4 + int(np.log(self.N) * 3)
        mu = int(lam/2)
        best = 100

        p_c = np.zeros(self.N)
        p_sigma = np.zeros(self.N)

        w = [1/mu for _ in range(mu)]
        mu_eff = 1 / sum([x*x for x in w])

        c_c = 2 / (self.N**2) #4/self.N, c1 in wiki
        c_cov = mu_eff/(self.N**2) # cmu in wiki
        c_sigma = 4 / self.N # cc in wiki

        d_sigma = 1 + sqrt(mu_eff/self.N)
        mu_cov = mu_eff

        start = time.time()
        for num in range(MAX_ITER):
            z_sample = np.random.multivariate_normal(mean, C, lam)
            z_sel = np.zeros(self.N)
            Z = np.zeros(self.N)
            D, B = np.linalg.eigh(C)
            # print(B)
            # D = np.sqrt(np.diag(D))

            sample = mean + self.step_size * z_sample

            score = [self.cost_func(s) for s in sample]

            selected = [[j,score[j]] for i, j in enumerate(np.argsort(score)) if i<mu]

            for s in selected:
                if s[1]<best:
                    best = s[1]
                    print(f"#{num} f({sample[s[0]]}) = {s[1]}")
                    print(f"{time.time() - start} s")
                z_sel = z_sel + (1/mu) * z_sample[s[0]]
                Z = Z + (1/mu) * z_sample[s[0]].reshape(self.N,1) *  z_sample[s[0]].reshape(1,self.N) 
            mean = mean + self.step_size * z_sel

            # norm_p_sigma = np.linalg.norm(p_sigma)
            # p_c = (1 - c_c) * p_c + (norm_p_sigma < (1.5 * sqrt(self.N))) * sqrt(1 - (1 - c_c)**2) * sqrt(mu_eff) * z_sel
            # C = (1 - c_cov) * C + c_cov/mu_cov * p_c.reshape(self.N,1) * p_c.reshape(1,self.N) + c_cov * (1 - 1/mu_cov) * Z
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(1 - (1 - c_sigma)**2)*sqrt(mu_eff) * B @ z_sel.reshape(self.N,1)
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # self.step_size = self.step_size * np.exp(c_sigma/d_sigma * (norm_p_sigma/self.N - 1))
            # print(z_sel)
            p_sigma = (1 - c_sigma) * p_sigma + sqrt(1 - (1 - c_sigma)**2)*sqrt(mu_eff) * B @ z_sel.reshape(self.N,1)
            norm_p_sigma = np.linalg.norm(p_sigma)
            # print(norm_p_sigma)
            p_c = (1 - c_c) * p_c + (norm_p_sigma < (1.5 * sqrt(self.N))) * sqrt(1 - (1 - c_c)**2) * sqrt(mu_eff) * z_sel
            C = (1 - c_cov) * C + c_cov/mu_cov * p_c.reshape(self.N,1) * p_c.reshape(1,self.N) + c_cov * (1 - 1/mu_cov) * Z
            self.step_size = self.step_size * np.exp(c_sigma/d_sigma * (norm_p_sigma/self.N - 1))
            print(self.step_size)
                    
        print(f"{num} DONE")