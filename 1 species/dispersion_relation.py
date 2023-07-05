#For the 1 species, 2D, nonlocal reaction-advection-diffusion system
#Class takes input of system parameters,
# and has a method outputting a value of the growth rate, lambda, for a given
# spatial mode, k. Also works with an array of ks, outputting an arry of lambdas
# the formula is based on linear stability analysis

import numpy as np
import scipy.special as sps

class dispersion():
    
    def __init__(self, k, rho, U, mu, xi, kernel_name):
        
        self.k = k
        self.rho = rho
        self.U = U
        self.mu = mu
        self.xi = xi
        
        # F(k*xi) = (k/xi)*H_1(k,xi), where H_1 is the canonical Hankel transform
        self.F = None
        # normalisation of kernel
        self.omega_0 = None
        
        if (kernel_name == "tophat"):
            self.omega_0 = 1/np.pi
            self.F = self.omega_0* -( sps.j0(self.xi*self.k) - sps.itj0y0(self.xi*self.k)[0]/(self.xi*self.k) )
            
        elif (kernel_name == "exponential"):
            self.omega_0 = 1/(2*np.pi)
            self.F = self.omega_0*((self.xi*self.k)**2)/np.power( (1 + (self.xi*self.k)**2), 3/2 )
            
        elif (kernel_name == "o3"):
            self.omega_0 = 1/(np.sqrt(2)*np.power(np.pi, 3/2))
            self.F = self.omega_0*((self.xi*self.k)**2)*np.exp(-0.5 * (self.xi*self.k)**2)
        
        else:
            raise AttributeError("Please enter \"tophat\", \"exponential\", or \"o3\" for the kernel name.")
            
    # See Equation (35) in the paper
    def dispersion_relation(self):
        return  ( -self.rho - (self.k**2) + 
                 (2*np.pi*self.U*(1-self.U)*self.mu/self.xi)*self.F )
    

if __name__ == '__main__':
    #Example of using this class.

    import matplotlib.pyplot as plt
    from matplotlib_style import matplotlib_style
    import matplotlib
    plt.rcParams.update(matplotlib_style(**{"text.usetex":True})) 
    use_PGF = False
    if (use_PGF):
        matplotlib.use('pgf')
        
    
    U = 0.5
    rho = 1
    mu = -5000
    xi = 1
    kernel_name = "tophat"
    k_points = np.linspace(0.01, 40, 1000)
    
    lambda_points = dispersion(k_points, rho, U, mu, xi, kernel_name).dispersion_relation()
     
    fig, ax = plt.subplots()
    ax.plot(k_points, lambda_points)
    ax.set_xlabel("k")
    ax.set_ylabel("$\lambda$")
    ax.axhline(y=0, c='black', linestyle='solid')
    
    #Use this for plotting wavenumbers (n_x, n_y) supported by a finite square 
    # domain with length L
    def k(n_x, n_y, L):
        k_finite = (2*np.pi/L)*np.sqrt(n_x**2 + n_y**2)
    
        print("k = %f"  %k_finite)
        
        plt.axvline(x=k_finite, c='black', linestyle='dashed')
        return 1
    
    if not use_PGF:
        plt.show()
    
    fig.savefig("dispersion_test.eps")