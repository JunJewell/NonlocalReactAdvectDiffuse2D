#For the 2 species, 2D, nonlocal reaction-advection-diffusion system
#Class takes input of system parameters,
# and has a method outputting a value of the growth rate, lambda, for a given
# spatial mode, k. Also works with an array of ks, outputting an arry of lambdas
# the formula is based on linear stability analys

import numpy as np
import scipy.special as sps

class dispersion():
    
    def __init__(self, k, U, V, D, mu_matrix, xi_matrix, kernel_name_matrix):
        
        #all matrices are 2x2 with indices 00=uu, 01=uv, 10=vu, 11=vv
        
        self.k = k
        self.U = U
        self.D = D
        self.V = V
        self.mu_matrix = mu_matrix
        self.xi_matrix = xi_matrix
        self.kernel_name_matrix = kernel_name_matrix
        
        #F(k*xi) = (k/xi)*H_1(k,xi), where H_1 is the canonical Hankel transform
        self.H_matrix = np.full(self.kernel_name_matrix.shape, None)
        # normalisation of kernel
        self.omega_0_matrix = np.full(self.kernel_name_matrix.shape, None)
        
        
        for (ij, kernel_name) in np.ndenumerate(self.kernel_name_matrix):
            
            if (kernel_name == "tophat"):
                self.omega_0_matrix[ij] = 1/np.pi
                self.H_matrix[ij] = self.omega_0_matrix[ij]* (self.xi_matrix[ij]/self.k) * -( sps.j0(self.xi_matrix[ij]*self.k) - sps.itj0y0(self.xi_matrix[ij]*self.k)[0]/(self.xi_matrix[ij]*self.k) )
                
            elif (kernel_name == "exponential"):
                self.omega_0_matrix[ij] = 1/(2*np.pi)
                self.H_matrix[ij] = self.omega_0_matrix[ij]*self.xi_matrix[ij]*((self.k)**3)/np.power( (1 + (self.xi_matrix[ij]*self.k)**2), 3/2 )
                
            elif (kernel_name == "o3"):
                self.omega_0_matrix[ij] = 1/(np.sqrt(2)*np.power(np.pi, 3/2))
                self.H_matrix[ij] = self.omega_0_matrix[ij]*self.xi_matrix[ij]*((self.k)**3)*np.exp(-0.5 * (self.xi_matrix[ij]*self.k)**2)
            
            else:
                raise AttributeError("Please enter \"tophat\", \"exponential\", or \"o3\" for the kernel name.")
                
        #integral transform of kernels with approporiate prefactors for each interaction
        # See equation 39 in the paper
        Gamma_uu = (2*np.pi*self.U*(1-self.U-self.V)*self.mu_matrix[0,0]/self.xi_matrix[0,0]**2)*self.H_matrix[0,0]
        Gamma_uv = (2*np.pi*self.U*(1-self.U-self.V)*self.mu_matrix[0,1]/self.xi_matrix[0,1]**2)*self.H_matrix[0,1]
        Gamma_vu = (2*np.pi*self.V*(1-self.U-self.V)*self.mu_matrix[1,0]/self.xi_matrix[1,0]**2)*self.H_matrix[1,0]
        Gamma_vv = (2*np.pi*self.V*(1-self.U-self.V)*self.mu_matrix[1,1]/self.xi_matrix[1,1]**2)*self.H_matrix[1,1]

        #from equation lambda^2 + C(k)*lambda + D(k)=0
        self.C_k = (self.k**2)*(1+self.D) - self.k*(Gamma_vv + Gamma_uu)
        
        self.D_k = ( self.D*(self.k**4) - (self.k**3)*(Gamma_vv + self.D*Gamma_uu) 
             + (self.k**2)*(Gamma_uu*Gamma_vv - Gamma_vu*Gamma_uv)
             )
    
    #See equation 40 in the paper
    def lambda_plus(self):
        #TAKING ON THE POSITIVE ROOT OF THE QUADRATIC EQUATION FOR LAMBDA
        #answers may be complex
        return 0.5* ( -self.C_k + np.emath.sqrt(self.C_k**2 - 4*self.D_k) )

    
    def lambda_minus(self):
        #TAKING ON THE NEGATIVE ROOT OF THE QUADRATIC EQUATION FOR LAMBDA
        #answers may be complex
        return 0.5* ( -self.C_k - np.emath.sqrt(self.C_k**2 - 4*self.D_k) )
    



if __name__ == '__main__':
    #Example of using this class
    
    import matplotlib.pyplot as plt
    from matplotlib_style import matplotlib_style
    import matplotlib
    plt.rcParams.update(matplotlib_style(**{"text.usetex":True})) 
    use_PGF = False
    if (use_PGF):
        matplotlib.use('pgf')
    
    U = 0.25
    V = 0.25
    D = 1
    xi_matrix = np.array([ [0.75,1], [1,0.75] ])
    mu_matrix = np.array([ [-2000,-1000], [1000, -2000] ])
    kernel_name_matrix = np.array([ ["tophat","tophat"], ["tophat","tophat"] ])
    
    theory_k_points = np.linspace(0.01, 100, 1000)
    dispersion_calculator = dispersion(theory_k_points, U, V, D, mu_matrix, xi_matrix, kernel_name_matrix)
    
    theory_lambda_plus_points = dispersion_calculator.lambda_plus()
    theory_lambda_minus_points = dispersion_calculator.lambda_minus()
    
    fig_disp, ax_disp = plt.subplots(1,1, figsize=(10,8))
    ax_disp.plot(theory_k_points, theory_lambda_plus_points.real, label="Re($\lambda$+)")

    ax_disp.axhline(y=0, c='black', linestyle='solid')
    ax_disp.set_xlabel("k")
    fontsize= 30
    ax_disp.set_ylabel("Re($\lambda$+)", fontsize=fontsize)
    ax_disp.legend(loc='upper left', fontsize=fontsize)

    ax_disp.set_ylim(-600,600)
    
    ax2 = ax_disp.twinx()
    ax2.set_ylabel("$\mathcal{C}(k)$", fontsize=fontsize)
    ax2.plot(theory_k_points, dispersion_calculator.C_k, label="$\mathcal{C}(k)$", color='green',linestyle='dashed')

    ax2.set_ylim(-1500, 1500)
    
    ax2.legend(loc="upper right", fontsize=fontsize)
    
    ax2.ticklabel_format(axis='y', style='sci', scilimits=[-3,3])
    
    ax_disp.set_xlim(-0.1, 17)
    
    plt.ticklabel_format(style='plain')
    
    #Use this for plotting wavenumbers (n_x, n_y) supported by a finite square 
    # domain with length L
    def k(n_x, n_y, L):
        k_finite = (2*np.pi/L)*np.sqrt(n_x**2 + n_y**2)
    
        print("k = %f"  %k_finite)
        
        plt.axvline(x=k_finite, c='black', linestyle='dashed')
        return 1
    
    if not use_PGF:
        plt.show()
    
    fig_disp.savefig("dispersion_test.eps")
            