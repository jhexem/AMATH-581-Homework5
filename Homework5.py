import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#------Problem 1------#

L = 10   #define initial conditions
m = 3
alpha = 0
n = 64
dt = 0.5
beta = 1
D1 = 0.1
D2 = 0.1

tvals = np.arange(0, 25+dt, dt)   #create time values
trange = [0, 25]
xyvals = np.linspace(-L, L, n)   #create meshgrid for the x and y values
X, Y = np.meshgrid(xyvals, xyvals)

A1 = X

Uvals = (np.tanh(np.sqrt(X ** 2 + Y ** 2)) - alpha) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X ** 2 + Y ** 2))   #define initial conditions
Vvals = (np.tanh(np.sqrt(X ** 2 + Y ** 2)) - alpha) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X ** 2 + Y ** 2))

A2 = Uvals

Uhat0 = fft2(Uvals)   #Fourier transform initial conditions
Vhat0 = fft2(Vvals)

A3 = np.real(Uhat0)

Uhat0vector = np.array([np.ndarray.flatten(Uhat0.T)]).T   #create the column vector form of Uhat0 and Vhat0
Vhat0vector = np.array([np.ndarray.flatten(Vhat0.T)]).T
UVhat0vector = np.vstack((Uhat0vector, Vhat0vector))   #stack the U and V column vectors into one large column vector

A4 = np.imag(UVhat0vector)

r1 = np.arange(0,n/2,1)   #generate k values
r2 = np.arange(-n/2,0,1)
kx = (2 * np.pi / L) * np.concatenate((r1,r2))
kx[0] = 10 ** (-6)
ky = kx.copy()
[KX, KY] = np.meshgrid(kx, ky)   #create the meshgrid for the kx and ky values

def RDsystemFFT(t, UVhatVector, beta, D1, D2, n, KX, KY):   #define the PDE
   UhatVector = UVhatVector[0:(n**2)]   #unstack the U hat and V hat vectors
   VhatVector = UVhatVector[(n**2):2*(n**2)]
   
   Uhat = UhatVector.reshape((n, n)).T   #reshape the U hat and V hat vectors into matrices
   Vhat = VhatVector.reshape((n, n)).T
   
   LapUhat = (-(KX ** 2) * Uhat) + (-(KY **2) * Uhat)   #define the Laplacian for both U hat and V hat
   LapVhat = (-(KX ** 2) * Vhat) + (-(KY **2) * Vhat)
   
   U = ifft2(Uhat)   #inverse Fourier transform to get U and V
   V = ifft2(Vhat)
   
   Uhat_t = Uhat - fft2(U ** 3) - fft2((V ** 2) * U) + fft2(beta * (U ** 2) * V) + fft2(beta * (V ** 3)) + D1 * LapUhat   #gets the Fourier transform of U_t from the PDE
   Vhat_t = -fft2(beta * (U ** 3)) - fft2(beta * (V ** 2) * U) - Vhat + fft2((U ** 2) * V) + fft2(V ** 3) + D2 * LapVhat   #gets the Fourier transform of V_t from the PDE
   
   newUhatVector = Uhat_t.T.reshape(n**2)   #reshape the Fourier transform of U_t and V_t into vectors
   newVhatVector = Vhat_t.T.reshape(n**2)
   
   return np.concatenate((newUhatVector, newVhatVector))   #return the single stacked vector for U_t and V_t

solFFT = integrate.solve_ivp(lambda t, UVhatVector: RDsystemFFT(t, UVhatVector, beta, D1, D2, n, KX, KY), trange, np.ndarray.flatten(UVhat0vector), t_eval=tvals)   #solve the PDE
solFFTy = solFFT.y

A5 = np.real(solFFTy)
A6 = np.imag(solFFTy)

UhatVectorSol = solFFTy[0:n**2, :]   #unstack all of the U hat and V hat vectors for each t value
VhatVectorSol = solFFTy[n**2:, :]

A7 = np.real(UhatVectorSol[4, :])

UhatSol = UhatVectorSol.reshape((n, n, len(tvals))).T   #reshape all of the U hat and V hat vectors to matrices for each t value
VhatSol = VhatVectorSol.reshape((n, n, len(tvals))).T

A8 = np.real(UhatSol)

A9 = np.real(ifft2(UhatSol[4, :, :]))

'''fig,ax = plt.subplots(subplot_kw = {"projection":"3d"}, figsize=(7, 7))   #plot the U solution at t=2
surf = ax.plot_surface(X, Y, A9, cmap='magma')
plt.xlabel('x')
plt.ylabel('y')
plt.show()   #'''

#------Problem 2------#

def cheb(N):
   # N is the number of points in the interior.
   if N==0:
      D = 0
      x = 1
      return D, x
   vals = np.linspace(0, N, N+1)
   x = np.cos(np.pi*vals/N)
   x = x.reshape(-1, 1)
   c = np.ones(N-1)
   c = np.pad(c, (1,), constant_values = 2)
   c *= (-1)**vals
   c = c.reshape(-1, 1)
   X = np.tile(x, (1, N+1))
   dX = X-X.T                  
   D  = (c*(1/c.T))/(dX+(np.eye(N+1)))       #off-diagonal entries
   D  = D - np.diag(sum(D.T))                #diagonal entries

   return D, x

m = 2   #define initial conditions
alpha = 1
n = 30

[D, x] = cheb(n)   #create Chebyshev differentiation matrix and Chebyshev points
x = x.reshape(n + 1)   #flatten Chebyshev points for the solver
D2matrix = D@D   #square differentiation matrix to get the second derivative matrix

x = x * L   #rescale the x points and the derivative
D2matrix = (1 / (L ** 2)) * D2matrix

D2matrix = D2matrix[1:-1, 1:-1]   #remove first row and column of the matrix to match boundary conditions
x2 = x[1:-1]   #remove first and last x points to match boundary conditions

y2 = x2.copy()   #create y values and generate X and Y meshgrids
[X, Y] = np.meshgrid(x2, y2)

tvals = np.arange(0, 25+dt, dt)   #generate t values
trange = [0, 25]

I = np.eye(len(D2matrix))   #generate the Laplacian matrix for Chebyshev
Lap = np.kron(D2matrix, I) + np.kron(I, D2matrix)

A10 = Lap
A11 = Y

U0vals = (np.tanh(np.sqrt(X ** 2 + Y ** 2)) - alpha) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X ** 2 + Y ** 2))   #define initial conditions
V0vals = (np.tanh(np.sqrt(X ** 2 + Y ** 2)) - alpha) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X ** 2 + Y ** 2))

A12 = V0vals

U0vector = np.array([np.ndarray.flatten(U0vals.T)]).T   #create the column vector form of U0 and V0
V0vector = np.array([np.ndarray.flatten(V0vals.T)]).T
UV0vector = np.vstack((U0vector, V0vector))   #stack the U and V column vectors into one large column vector

A13 = UV0vector

def RDsystemCheb(t, UVvector, Lap, n, beta, D1, D2):
   U = UVvector[0:((n-1)**2)]   #unstack the U and V vectors
   V = UVvector[((n-1)**2):2*((n-1)**2)]
   
   U_t = U - (U ** 3) - ((V ** 2) * U) + (beta * (U ** 2) * V) + (beta * (V ** 3)) + (D1 * (Lap @ U))   #gets U_t from the PDE
   V_t = (-beta * (U ** 3)) + (-beta * (V ** 2) * U) - V + ((U ** 2) * V) + (V ** 3) + (D2 * (Lap @ V))   #gets V_t from the PDE
   
   return np.concatenate((U_t, V_t))   #return the single stacked vector for U_t and V_t

solCheb = integrate.solve_ivp(lambda t, UVvector: RDsystemCheb(t, UVvector, Lap, n, beta, D1, D2), trange, np.ndarray.flatten(UV0vector), t_eval=tvals)
solChebY = solCheb.y

A14 = solChebY.T

UvectorSol = solChebY[0:(n-1)**2, :]   #unstack all of the U and V vectors for each t value
VvectorSol = solChebY[(n-1)**2:, :]

A15 = VvectorSol[:, 4]

Usol = UvectorSol.reshape((n-1, n-1, len(tvals))).T   #reshape all of the U and V vectors to matrices for each t value
Vsol = VvectorSol.reshape((n-1, n-1, len(tvals))).T

Utsol = Usol[4, :, :]   #get the U and V solutions at t=2
Vtsol = Vsol[4, :, :]

Utsol = np.pad(Utsol, [1,1])
Vtsol = np.pad(Vtsol, [1,1])

A16 = Utsol

y = x.copy()   #create y values and generate X and Y meshgrids
[X, Y] = np.meshgrid(x, y)

'''fig,ax = plt.subplots(subplot_kw = {"projection":"3d"}, figsize=(7, 7))   #plot the U solution at t=2
surf = ax.plot_surface(X, Y, A16, cmap='magma')
plt.xlabel('x')
plt.ylabel('y')
plt.show()   #'''