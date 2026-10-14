"""Explicit conventional baselines and a local factorization from the paper."""
import numpy as np


def _validate_length(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError('positive integer transform length required')


def _input(x, n):
    values = np.asarray(x, dtype=np.complex128)
    if values.ndim != 1 or len(values) != n or not np.all(np.isfinite(values)):
        raise ValueError('finite 1D input of the planned length required')
    return values


class DirectPlan:
    def __init__(self, n):
        _validate_length(n)
        self.n = n
        indices = np.arange(n)
        self.matrix = np.exp((-2j*np.pi/n)*(indices[:, None]*indices[None, :]))

    def apply(self, x):
        return self.matrix @ _input(x, self.n)

    def counts(self):
        return {'complex_scalar_multiplications': self.n**2,
                'complex_add_subtract': self.n*(self.n-1)}


class Radix2Plan:
    def __init__(self, n):
        _validate_length(n)
        if n & (n-1):
            raise ValueError('radix-2 requires power-of-two length')
        self.n, self.k = n, n.bit_length()-1
        self.reversal = np.array([int(f'{i:0{self.k}b}'[::-1], 2) if self.k else 0
                                  for i in range(n)])
        self.twiddles = [np.exp(-2j*np.pi*np.arange(m//2)/m)
                         for m in (1 << stage for stage in range(1, self.k+1))]

    def apply(self, x):
        output = _input(x, self.n)[self.reversal].copy()
        for twiddle in self.twiddles:
            half = len(twiddle)
            blocks = output.reshape(-1, 2*half)
            left = blocks[:, :half].copy()
            right = blocks[:, half:]*twiddle
            blocks[:, :half] = left+right
            blocks[:, half:] = left-right
        return output

    def counts(self):
        return {'complex_scalar_multiplications': self.n*self.k//2,
                'complex_add_subtract': self.n*self.k}


def radix2_fft(x):
    return Radix2Plan(len(x)).apply(x)


class NewtonPlan:
    """F=N diag(N_ii)^(-1) N^T; NOT the paper's all-length fast compiler."""
    def __init__(self, n):
        _validate_length(n)
        if n > 256:
            raise ValueError('exploratory Newton plan limited to n <= 256')
        self.n = n
        u = np.exp(-2j*np.pi/n)
        H = np.ones(n, dtype=np.complex128)
        for j in range(1, n):
            H[j] = H[j-1]*(1-u**j)
        self.N = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for k in range(i+1):
                self.N[i, k] = (-1)**k*u**(k*(k-1)//2)*H[i]/H[i-k]
        self.inverse_diagonal = 1/self.N.diagonal()

    def apply(self, x):
        x = _input(x, self.n)
        middle = np.array([self.N[i:, i] @ x[i:] for i in range(self.n)])
        middle *= self.inverse_diagonal
        return np.array([self.N[i, :i+1] @ middle[:i+1] for i in range(self.n)])

    def counts(self):
        return {'complex_scalar_multiplications': self.n*(self.n+1)+self.n,
                'complex_add_subtract': self.n*(self.n-1)}

    def diagnostics(self):
        return {'N_condition_2': float(np.linalg.cond(self.N)),
                'N_max_coefficient_abs': float(np.max(np.abs(self.N))),
                'inverse_diagonal_max_abs': float(np.max(np.abs(self.inverse_diagonal)))}


def tensor_c(x, optimized=True):
    """C tensor powers, a different transform from DFT. No speedup claim."""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    if x.ndim != 1 or not n or n & (n-1) or not np.all(np.isfinite(x)):
        raise ValueError('finite power-of-two 1D array required')
    out = x.copy()
    a, b = (1+1j)/2, (1-1j)/2
    for bit in range(n.bit_length()-1):
        half = 1 << bit
        blocks = out.reshape(-1, 2*half)
        left, right = blocks[:, :half].copy(), blocks[:, half:].copy()
        if optimized:
            t = b*(right-left)
            blocks[:, :half], blocks[:, half:] = left+t, right-t
        else:
            blocks[:, :half], blocks[:, half:] = a*left+b*right, b*left+a*right
    return out
