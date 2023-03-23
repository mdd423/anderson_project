import numpy as np
import scipy.sparse as sparse

class RandomMatDist:
    def __init__(self,f,u):
        self.f = f
        self.u = u

    def sample(self,shape):
        @np.vectorize
        def _transform(x):
            if x < self.f:

                return (1 - self.u/2) + (self.u*x)/self.f
            if x >= self.f:

                return (-1 - self.u/2) + (self.u*(x-self.f))/(1-self.f)

        x = np.random.rand(*shape)
        return _transform(x)

def get_matrix(N,f,u,g):
    dist = RandomMatDist(f,u)
    ssss = dist.sample((N,2))
    inds = np.arange(0,N)
    mat1 = sparse.csr_matrix((ssss[:,0] * np.exp(g), (inds, (inds+1) % N)), shape=(N, N))
    mat2 = sparse.csr_matrix((ssss[:,1] *np.exp(-g), ((inds+1) % N, inds)), shape=(N, N))
    return mat1 + mat2

class BetaPrimeDist:
    def __init__(self,beta,w):
        self.beta = beta
        self.w    = w

    def sample(self,shape):
        x = (np.random.rand(*shape) * self.w) + (self.beta - self.w/2)
        return x

def get_J_matrix(N,u,w,g=0.0,f=1.0,alpha=1.0,beta=1.0,gamma=1.0):
    dist = RandomMatDist(f=f,u=u)
    ssss = dist.sample((N,2))
    inds = np.arange(0,N)
    mat1 = sparse.csr_matrix((ssss[:,0] * np.exp(g), (inds, (inds+1) % N)), shape=(N, N))
    mat2 = sparse.csr_matrix((ssss[:,1] *np.exp(-g), ((inds+1) % N, inds)), shape=(N, N))
    neig = alpha * (mat1 + mat2)

    idnt = gamma * sparse.eye(N)

    betp = BetaPrimeDist(beta=beta,w=w)
    inhb = betp.sample((N,N))
    # inhb[inds,inds] = 0.0

    return neig + idnt - inhb
