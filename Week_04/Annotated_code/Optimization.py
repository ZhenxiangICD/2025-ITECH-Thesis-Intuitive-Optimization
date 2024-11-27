import math
import numpy as np
import tensorflow as tf

from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from Derivatives import *

@tf.function
def objGradPsiSum(n_tiling_params, inputs, ti, model):
    """
    Computes:
        1. psi: Total elastic potential, summed over the batch.
        2. grad: Gradient of the total elastic potential w.r.t. input strain tensor: ∂Φ/∂E.

    Inputs:
        - n_tiling_params: Number of tiling parameters (int).
        - inputs: Flattened strain tensor inputs for the batch (shape: [batch_dim * 3]).
        - ti: Tiling parameters (shape: [n_tiling_params]).
        - model: Neural network model to compute energy potential Φ(T, E).

    Outputs:
        - psi: Summed elastic potential (scalar).
        - grad: Gradient of elastic potential w.r.t. input strain tensor (shape: [batch_dim * 3]).

    Links to Equations in the Paper:
        - Equation (8): Φ(T_i, E_i, θ): Elastic potential.
        - Gradient computation: ∂Φ(T_i, E_i, θ)/∂E_i.
    """
    batch_dim = int(inputs.shape[0] // 3)
    
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        ti_batch = tf.tile(tf.expand_dims(ti, 0), (batch_dim, 1))
        strain = tf.reshape(inputs, (batch_dim, 3))
        nn_inputs = tf.concat((ti_batch, strain), axis=1)
        psi = model(nn_inputs, training=False)
        psi = tf.math.reduce_sum(psi, axis=0)
    grad = tape.gradient(psi, inputs)  # Compute gradient of total potential w.r.t. input strain tensor.
    del tape
    return tf.squeeze(psi), tf.squeeze(grad) # Return scalar psi and gradient.

@tf.function
def hessPsiSum(n_tiling_params, inputs, ti, model):
    """
    Computes:
        1. Hessian of the summed elastic potential w.r.t. strain inputs: ∂²Φ/∂E².

    Inputs:
        - n_tiling_params: Number of tiling parameters (int).
        - inputs: Flattened strain tensor inputs for the batch (shape: [batch_dim * 3]).
        - ti: Tiling parameters (shape: [n_tiling_params]).
        - model: Neural network model to compute energy potential Φ(T, E).

    Outputs:
        - hess: Hessian matrix of the elastic potential w.r.t. strain inputs (shape: [batch_dim * 3, batch_dim * 3]).

    Links to Equations in the Paper:
        - Equation (11): Stiffness tensor C_α = ∂²Φ(T, E)/∂E².
    """
    batch_dim = int(inputs.shape[0] // 3) # Number of strain samples in the batch.
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs) # Monitor inputs for second derivative computation.
        with tf.GradientTape() as tape:
            tape.watch(inputs)  # Monitor inputs for first derivative computation.
            ti_batch = tf.tile(tf.expand_dims(ti, 0), (batch_dim, 1))  # Tile ti for batch compatibility.
            strain = tf.reshape(inputs, (batch_dim, 3)) # Reshape inputs into strain vectors.
            nn_inputs = tf.concat((ti_batch, strain), axis=1) # Combine tiling and strain for NN input.
            psi = model(nn_inputs, training=False) # Compute energy potential Φ(T, E).
            psi = tf.math.reduce_sum(psi, axis=0)
        grad = tape.gradient(psi, inputs) # Compute gradient of potential w.r.t. input strain tensor E
    hess = tape_outer.jacobian(grad, inputs) # Compute Hessian of potential w.r.t. input starin tensor E
    del tape
    del tape_outer
    return tf.squeeze(hess)

def optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
    theta, strains, tiling_params, verbose = True):

    """
    Performs OUTER optimization for energy density(output of this optimization model) and strain tensor(variable of this optimization model),
    
    given 1. tiling parameters(treated as fixed input) ; 2. a list of target strain magnitute ; 3. SINGLE uniaxial loading direction.

    Computes:
        1. Optimal strain tensor for each direction θ given a strain magnitude.
        2. Constrained optimization of Φ(T, E) with uniaxial constraint.

    Inputs:
        - model: Neural network model.
        - n_tiling_params: Number of tiling parameters (int).
        - theta: Direction of uniaxial strain (scalar, in radians).
        - strains: List of strain magnitudes for optimization.
        - tiling_params: Tiling parameters (shape: [n_tiling_params]).
        - verbose: Display optimization details (bool).

    Outputs:
        - Reshaped optimized strains (shape: [len(strains), 3]).

    Links to Equations in the Paper:
        - Equation (9c): Inner optimization to satisfy uniaxial strain constraints.
    """

    d = np.array([np.cos(theta), np.sin(theta)]) # Directional vector for uniaxial loading.
    strain_init = [] # Initialize strain vectors.
    for strain in strains: 
        strain_tensor_init = np.outer(d, d) * strain # Create strain tensor in the specified direction.
        strain_init.append(np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]]))

    strain_init = np.array(strain_init).flatten() # Flatten into a single array.
    
    m = len(strain_init) // 3  # Number of strain samples
    n = len(strain_init) # Total number of variables.
    A = np.zeros((m, n))
    lb = []
    ub = []

    for i in range(m):
        A[i, i * 3:i * 3 + 3] = computedCdE(d) #Fill constraints using directional derivatives.
        lb.append(strains[i])
        ub.append(strains[i])

    uniaxial_strain_constraint = LinearConstraint(A, lb, ub)  # Define linear constraint.

    def hessian(x):
        """
        Computes Hessian of Φ(T, E) for trust-region optimization.
        """        
        H = hessPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        H = H.numpy()
        
        ev_H = np.linalg.eigvals(H) # Ensure positive-definiteness.
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev))
        return H

    def objAndEnergy(x):
        """
        Computes objective Φ(T, E) and its gradient for optimization.
        """
        obj, grad = objGradPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        
        obj = obj.numpy()
        grad = grad.numpy().flatten()
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True,
         hess=hessian,
            constraints=[uniaxial_strain_constraint],
            options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, 
            hess=hessian,
            constraints= [uniaxial_strain_constraint],
            options={'disp' : False})
    
    return np.reshape(result.x, (m, 3)) # Reshape the result into strain vectors.



def computeUniaxialStrainThetaBatch(n_tiling_params, strain, 
    thetas, model, tiling_params, verbose = True):

    """
    Performs OUTER optimization for energy density(model output) and strain tensor(optimal model variable),
    
    given 1. tiling parameters(treated as fixed input) ; 2. one shared target starin magnitute ; 3. MULTIPLE uniaxial loading directions.

    Computes:
        1. Optimal strain tensors for multiple directions (thetas) given a strain magnitude.
        2. Solves constrained optimization to satisfy uniaxial strain constraints for each direction.

    Inputs:
        - n_tiling_params: Number of tiling parameters (int).
        - strain: Prescribed strain magnitude (scalar).
        - thetas: Array of directions for uniaxial loading (shape: [n_angles]).
        - model: Neural network model for energy computation.
        - tiling_params: Tiling parameters (shape: [n_tiling_params]).
        - verbose: Display optimization details (bool).

    Outputs:
        - Optimized strain tensors for each direction (shape: [len(thetas), 3]).

    Links to Equations in the Paper:
        - Equation (9c): Inner optimization under directional strain constraints.
    """    
    strain_init = []
    for theta in thetas:
        d = np.array([np.cos(theta), np.sin(theta)]) # Directional vector for uniaxial loading.
        strain_tensor_init = np.outer(d, d) * strain # Create strain tensor in the specified direction.
        strain_init.append(np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]]))

    strain_init = np.array(strain_init).flatten() # Flatten multiple strain tensors into a single array.
    
    m = len(strain_init) // 3
    n = len(strain_init)
    A = np.zeros((m, n))
    lb = []
    ub = []

    for i in range(m):
        d = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
        A[i, i * 3:i * 3 + 3] = computedCdE(d)  # Fill constraints using directional derivatives.
        lb.append(strain) # Set lower bound.
        ub.append(strain) # Set upper bound as same.

    uniaxial_strain_constraint = LinearConstraint(A, lb, ub)  # Define linear equality constraint.

    def hessian(x):
        """
        Computes PSD Hessian of Φ(T, E) w.r.t. E for trust-region optimization.
        """        
        H = hessPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model)
        H = H.numpy()
        
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-6))
        return H

    def objAndEnergy(x):
        """
        Computes objective Φ(T, E) and its gradient w.r.t. E for optimization.
        """        
        obj, grad = objGradPsiSum(n_tiling_params, tf.convert_to_tensor(x), 
            tf.convert_to_tensor(tiling_params), model) 
        
        obj = obj.numpy()
        grad = grad.numpy().flatten()
        return obj, grad

    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints=[uniaxial_strain_constraint],
            options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, 
            hess=hessian,
            constraints= [uniaxial_strain_constraint],
            options={'disp' : False})
    
    return np.reshape(result.x, (m, 3)) # Reshape the result into strain vectors.

@tf.function
def valueGradHessian(n_tiling_params, inputs, model):
    """
    Computes:
        1. ψ: Elastic potential Φ(T, E).
        2. stress: Stress tensor S = ∂Φ/∂E (Voigt notation, shape: [batch_dim, 3]).
        3. C: Stiffness tensor C = ∂²Φ/∂E² (shape: [batch_dim, 3, 3]).

    Inputs:
        - n_tiling_params: Number of tiling parameters (int).
        - inputs: Tiling parameters and strain concatenated (shape: [batch_dim, n_tiling_params + 3]).
        - model: Neural network model for energy computation.

    Outputs:
        - psi: Elastic potential (shape: [batch_dim]).
        - stress: Stress tensor (shape: [batch_dim, 3]).
        - C: Stiffness tensor (shape: [batch_dim, 3, 3]).

    Links to Equations in the Paper:
        - Equation (11): Stiffness tensor C_α = ∂²Φ(T, E)/∂E².
        - Stress derivation: S = ∂Φ/∂E.
    """    
    batch_dim = inputs.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            psi = model(inputs, training=False)
            dedlambda = tape.gradient(psi, inputs)
            
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    del tape
    del tape_outer
    return psi, stress, C

@tf.function
def computeDirectionalStiffness(n_tiling_params, inputs, thetas, model):
    """
    Computes:
        1. Directional stiffness profile for given tiling parameters and strain directions.

    Inputs:
        - n_tiling_params: Number of tiling parameters (int).
        - inputs: Concatenated tiling parameters and strain tensors (shape: [batch_dim, n_tiling_params + 3]).
        - thetas: Array of angles for directional stiffness evaluation (shape: [batch_dim]).
        - model: Neural network model for energy computation.

    Outputs:
        - stiffness: Directional stiffness values for each angle (shape: [batch_dim]).

    Links to Equations in the Paper:
        - Equation (10a): Directional stiffness computation.
        - Stiffness tensor inversion: S_α : C_α = I.
    """    
    thetas = tf.expand_dims(thetas, axis=1)

    # Compute directional vector in Voigt notation for each theta.
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)
    psi, stress, C = valueGradHessian(n_tiling_params, inputs, model) # Compute energy, stress, and stiffness.
    
    # Initialize stiffness computation.
    Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :]) # Solve Sd = C⁻¹d for the first sample.
    dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0) # Compute dᵀ(C⁻¹)d.    
    stiffness = tf.divide(tf.constant([1.0], dtype=tf.float64), dTSd) # Compute stiffness as (dᵀ(C⁻¹)d)⁻¹.

    # Loop over remaining batch samples to compute directional stiffness.
    for i in range(1, C.shape[0]):
        
        Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
        dTSd = tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0)
        stiffness = tf.concat((stiffness, tf.divide(tf.constant([1.0], dtype=tf.float64), dTSd)), 0)
    return tf.squeeze(stiffness) # Return directional stiffness values.


def optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, 
    theta, strain, tiling_params, verbose = True):
    """
    Performs INNER optimization of strain E given a direction θ, strain magnitude, and tiling parameters T.

    Computes:
        1. optimized strain tensor 
        2. Sensitivity analysis to tiling parameters for use in outer optimization.

    Inputs:
        - model: Neural network model for energy computation (Φ).
        - n_tiling_params: Number of tiling parameters (int).
        - theta: Angle defining the loading direction (scalar).
        - strain: Target strain magnitude (scalar).
        - tiling_params: Tiling parameters T (shape: [n_tiling_params], treated not as variables).
        - verbose: Display optimization details (bool).

    Outputs:
        - result.x: Optimized strain tensor E (shape: [3]).
        - dqdp: Sensitivity of strain with respect to tiling parameters T, ∂E/∂T (shape: [4, n_tiling_params]).

    Links to Equations in the Paper:
        - Equation (9c): Inner optimization for strain magnitude constraints.
        - Equation (17)-(18): Sensitivity computation using total derivatives.
    """    

    # Initialize strain tensor (Voigt notation).
    strain_init = np.array([0.105, 0.2, 0.01])

    d = np.array([np.cos(theta), np.sin(theta)]) # Directional vector d.

    # Construct initial guess for strain tensor in the specified direction.
    strain_tensor_init = np.outer(d, d) * strain
    strain_init = np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]])

    def constraint(x):
        """
        Constraint: Ensure strain magnitude matches the prescribed value.
        Computes dᵀE d - strain.
        """        
        strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2)) # Reshape Voigt to 2x2.
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d))) # Compute dᵀE d.
        c = dTEd - strain
        return c

    def hessian(x):
        """
        Hessian computation for optimization.
        Links to Equation (11): ∂²Φ(T, E)/∂E² = C.
        """        
        model_input = tf.convert_to_tensor([np.hstack((tiling_params, x))])  # Combine tiling parameters and strain as input to NN.
        C = computeStiffnessTensor(n_tiling_params, model_input, model) # Compute stiffness tensor C.
        H = C.numpy() # Convert to NumPy array.
        ev_H = np.linalg.eigvals(H) # Ensure Hessian is positive definite.
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-6))
        return H

    def objAndEnergy(x):
        """
        Objective and gradient computation for optimization.
        Links to Equation (8): Compute Φ(E, T) and ∂Φ(E, T)/∂E.
        """        
        model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, x))))]) 
        _, stress, _, psi = testStep(n_tiling_params, model_input, model) # Compute Φ(E, T) and stress S.
        
        obj = np.squeeze(psi.numpy())  # Scalar energy value.
        grad = stress.numpy().flatten() # Gradient of energy w.r.t. strain.
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints={"fun": constraint, "type": "eq"},
            options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints={"fun": constraint, "type": "eq"},
            # options={'disp' : False})
        options={'disp' : False, 'maxiter':100, 'gtol' : 1e-5})
    
    # Compute sensitivity derivatives ∂E/∂T using implicit function therom. 
    # VERY IMPORTANT TO BE FULLY UNDERSTOOD!!!!
    opt_model_input = tf.convert_to_tensor([np.hstack((tiling_params, result.x))]) # Combine optimized strain to tiling params as NN input.
    
    d2Phi_dE2 = computeStiffnessTensor(n_tiling_params, opt_model_input, model) # Compute ∂²Φ/∂E².
    dCdE = computedCdE(d) # Compute directional derivative dᵀCd.

    # Assemble the sensitivity matrix.
    # The Lagrangian of the model is L(q),  q = [E, λ]
    # In optimal state, g := ∂L/∂q = 0 
    # To compute sensitivity dq/dp, we need to compute ∂g/∂q = ∂²L/∂q², ∂g/∂p = ∂²L/∂q∂p. Here `p = T` (tiling parameters).

    # The shape of ∂²L/∂q∂p is (4, n_tiling_params) , where 4 = 3 x ∂Φ/∂E terms + 1 x constraint term
    # ∂Φ/∂E = S, we use existing function computedStressdp to compute its derivative w.r.t. p 
    # constraint term should stay zero in optimal state, d2Ldqdp[3, :] = 0
    d2Ldqdp = np.zeros((3 + 1, n_tiling_params))  # Matrix to hold ∂²L/∂q∂p
    dsigma_dp = computedStressdp(n_tiling_params, opt_model_input, model) #∂S/∂p .

    # If `dsigma_dp` is a 1D array, reshape it for compatibility.
    if (len(dsigma_dp.shape) == 1):
        dsigma_dp = np.reshape(dsigma_dp, (3, 1))
    d2Ldqdp[:3, :] = dsigma_dp # Top-left block corresponds to ∂S/∂p.

    # The shape of ∂²L/∂q² (4, 4)
    # Top-left 3x3 block is taking derivative of ∂Φ/∂E = S w.r.t. E, which is the stiffness tensor we computed already as d2Phi_dE2
    # Top right and bottom left block is the derivative of constraint term w.r.t. E, which is directional derivative we computed already as dCdE
    d2Ldq2 = np.zeros((3 + 1, 3 + 1))
    d2Ldq2[:3, :3] = d2Phi_dE2 # Top-left block is ∂²Φ/∂E² 
    d2Ldq2[:3, 3] = -dCdE # Top-right block: Cross-derivative of Lagrangian w.r.t. E and λ.
    d2Ldq2[3, :3] = -dCdE # Bottom-left block: Symmetric to top-right.
    
    # Perform LU decomposition to solve the linear system ∂²L/∂q² dq/dp = -∂²L/∂q∂p.
    lu, piv = lu_factor(d2Ldq2)
    
    dqdp = lu_solve((lu, piv), -d2Ldqdp) # # Solve for ∂q/∂p = [∂E/∂T, ∂λ/∂T].

    
    return result.x, dqdp


def optimizeUniaxialStrainSingleDirection(model, n_tiling_params, 
    theta, strain, tiling_params, verbose = False):

    """
    Simplified version of `optimizeUniaxialStrainSingleDirectionConstraint`.
    Focuses on optimizing strain E without computing sensitivity derivatives.
    Please refer to the previous annotation

    Inputs:
        - model: Neural network model for energy computation.
        - n_tiling_params: Number of tiling parameters (int).
        - theta: Angle defining the loading direction (scalar).
        - strain: Target strain magnitude (scalar).
        - tiling_params: Tiling parameters T (shape: [n_tiling_params]).
        - verbose: Display optimization details (bool).

    Outputs:
        - result.x: Optimized strain tensor E (shape: [3]).

    Links to Equations in the Paper:
        - Equation (9c): Inner optimization for strain magnitude constraints.
    """   
    
    # Same initialization and structure as `optimizeUniaxialStrainSingleDirectionConstraint`,
    # but without sensitivity derivative computations.
    d = np.array([np.cos(theta), np.sin(theta)])
    strain_tensor_init = np.outer(d, d) * strain
    strain_init = np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]])
    
    def constraint(x):
        strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        c = dTEd - strain
        return c


    def hessian(x):
        model_input = tf.convert_to_tensor([np.hstack((tiling_params, x))])
        C = computeStiffnessTensor(n_tiling_params, model_input, model)
        H = C.numpy()
        ev_H = np.linalg.eigvals(H)
        min_ev = np.min(ev_H)
        if min_ev < 0.0:
            H += np.diag(np.full(len(x),min_ev + 1e-7))
        return H

    def objAndEnergy(x):
        model_input = tf.convert_to_tensor([np.hstack((np.hstack((tiling_params, x))))])
        _, stress, _, psi = testStep(n_tiling_params, model_input, model)
        
        obj = np.squeeze(psi.numpy()) 
        grad = stress.numpy().flatten()
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    if verbose:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints={"fun": constraint, "type": "eq"},
            options={'disp' : True, 'maxiter':100, 'gtol' : 1e-6})
            # options={'disp' : True})
    else:
        result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
            constraints={"fun": constraint, "type": "eq"},
            options={'disp' : False, 'maxiter':100, 'gtol' : 1e-5})
        # result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, 
        #     constraints={"fun": constraint, "type": "eq"},
        #     options={'disp' : False})
    
    
    return result.x