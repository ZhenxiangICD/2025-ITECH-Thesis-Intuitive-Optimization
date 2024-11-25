import math
import numpy as np
import tensorflow as tf

@tf.function
def testStep(n_tiling_params, lambdas, model):
    
    """
    Computes:
        1. dstress_dp： Sensitivity of stress to tiling parameters: ∂S/∂T.
        2. stress： Stress: S = ∂Φ/∂E (Voigt stress tensor, representing 2x2 tensor as 1x3 vector).
        3. de_dp： Sensitivity of elastic potential to tiling parameters: ∂Φ/∂T.
        4. elastic_potential： Elastic potential: Φ(T, E).

    Links to Equation (8) in the paper:
        - Φ(T_i, E_i, θ): Elastic energy potential returned by the model.
        - S_i = ∂Φ(T_i, E_i, θ)/∂E_i: Stress derived via backpropagation.
        - Used in the inner optimization problem (Equation 9c).
    """
         
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)

            # Φ(T, E), the elastic energy potential:
            elastic_potential = model(lambdas, training=False)

            # dedlambda = ∂Φ/∂λ -> Gradient w.r.t. inputs, λ = [T, E]:
            dedlambda = tape.gradient(elastic_potential, lambdas)

            # Parse outputs into energy and stress components:
            batch_dim = elastic_potential.shape[0]

            #stress = ∂Φ/∂E = S
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])

            #de_dp = ∂Φ/∂T
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    
    #dstress_dp = ∂S/∂T
    dstress_dp = tape_outer.batch_jacobian(stress, lambdas)[:, :, 0:n_tiling_params]

    del tape
    del tape_outer
    return dstress_dp, stress, de_dp, elastic_potential

@tf.function
def testStepd2edp2(n_tiling_params, lambdas, model):
    """
    Computes:
        1. Hessian of elastic potential: ∂²Φ/∂T².
        2. de_dp：Gradient of elastic potential: ∂Φ/∂T.
        3. elastic_potential：Elastic potential: Φ(T, E).

    Links to Equation (8):
        - First term: Φ(T, E, θ) - Elastic potential.
        - Second term: Gradient and Hessian terms required for sensitivity analysis (outer optimization in Equation 9a).
    """
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)

            # Compute Φ(T, E):
            elastic_potential = model(lambdas, training=False)

            # dedlambda = ∂Φ/∂λ，λ = [T, E]:
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]

            # Extract sensitivity terms w.r.t. tiling parameters
            # de_dp = ∂Φ/∂T: 
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    
    # ∂²Φ/∂T² (Hessian of Φ w.r.t. tiling parameters T):
    d2edp2 = tape_outer.batch_jacobian(de_dp, lambdas)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return d2edp2, de_dp, elastic_potential

@tf.function
def computedStressdp(n_tiling_params, opt_model_input, model):
    """
    Computes:
        - Sensitivity of stress (S) to tiling parameters (T), ∂S/∂T.

    Links to Equation (18):
        - Appears in the Hessian matrix of the Lagrangian system (d²L/dqdp).
        - Required for gradient-based sensitivity analysis in outer optimization.
    """

    #the steps and variables are identical to previous functions but only ouput dStressdp
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(opt_model_input)
        with tf.GradientTape() as tape:
            tape.watch(opt_model_input)
            
            elastic_potential = model(opt_model_input, training=False)
            dedlambda = tape.gradient(elastic_potential, opt_model_input)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    dstress_dp = tape_outer.batch_jacobian(stress, opt_model_input)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer
    return tf.squeeze(dstress_dp)

@tf.function
def computeStiffnessTensor(n_tiling_params, inputs, model):
    """
    Computes:
        - Stiffness tensor (C): ∂²Φ/∂E².

    Links to Equation (11):
        - Cα = ∂²Φ(T, E)/∂E².
        - Part of the compliance tensor inversion (Sα : Cα = I).
    """

    batch_dim = inputs.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            
            psi = model(inputs, training=False)
            dedlambda = tape.gradient(psi, inputs)
            
            #stress = ∂Φ/∂E = S, slice gradient excluding T
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
    
    #C = ∂S/∂E = ∂²Φ/∂E², slicing stress jacobian excluding T
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    del tape_outer
    del tape
    return tf.squeeze(C)


def computedCdE(d):
    """
    Computes directional derivatives of the stiffness tensor C along the direction d. 

    Inputs:
        - d: Direction vector, typically \( d = [\cos(\theta), \sin(\theta)] \).

    Outputs:
        - dTd [d_x^2, d_y^2, 2 * d_x * d_y] (Voigt components).
    """

    _i_var = np.zeros(7)
    _i_var[0] = (d[1])*(d[0])
    _i_var[1] = (d[0])*(d[1])
    _i_var[2] = 0.5
    _i_var[3] = (_i_var[1])+(_i_var[0])
    _i_var[4] = (d[0])*(d[0])
    _i_var[5] = (d[1])*(d[1])
    _i_var[6] = (_i_var[3])*(_i_var[2])
    return np.array(_i_var[4:7])

@tf.function
def computedPsidEEnergy(n_tiling_params, model_input, model):
    """
    Computes:
        1. stress: Stress S = ∂Φ/∂E (Voigt stress tensor, representing 2x2 tensor as 1x3 vector).

    Links to Equation (8) in the paper:
        - Φ(T_i, E_i, θ): Elastic energy potential returned by the model.
        - S_i = ∂Φ(T_i, E_i, θ)/∂E_i: Stress derived via backpropagation.
        - Used in the inner optimization problem (Equation 9c).

    Inputs:
        - n_tiling_params: Number of design parameters.
        - model_input: Combined tiling parameters T and strain E.
        - model: Neural network representation of Φ(T, E).

    Outputs:
        - stress: Voigt representation of the stress tensor S (shape: batch_dim x 3).
    """    
    with tf.GradientTape() as tape:
        tape.watch(model_input)# Watch model input for gradient computation.
        psi = model(model_input, training=False)    # Evaluate energy density Φ(T, E).
        dedlambda = tape.gradient(psi, model_input) # Compute gradient ∂Φ/∂λ, λ = [T, E].
        batch_dim = psi.shape[0]    # Determine batch size.
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])  # Extract Voigt stress.
    del tape# Clean up the gradient tape.
    return tf.squeeze(stress) # Return stress tensor.

@tf.function
def computedPsidEGrad(n_tiling_params, inputs, model):
    """
    SAME AS computeStiffnessTensor
    """

    batch_dim = inputs.shape[0]# Determine batch size.
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)# Watch inputs for second-order derivative.
        with tf.GradientTape() as tape:
            tape.watch(inputs)  # Watch inputs for first-order derivative.
            
            psi = model(inputs, training=False)  # Evaluate energy density Φ(T, E).
            dedlambda = tape.gradient(psi, inputs)
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
    
    #C = ∂S/∂E = ∂²Φ/∂E², slicing stress jacobian excluding T
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    del tape
    del tape_outer
    return tf.squeeze(C)

@tf.function
def psiValueGradHessian(n_tiling_params, inputs, model):
    """
    Computes:
        1. energy density: Φ(T, E).
        2. stress: S = ∂Φ/∂E (Voigt stress tensor, representing 2x2 tensor as 1x3 vector).
        3. stiffness tensor: C = ∂²Φ/∂E² (second derivative of energy w.r.t strain).

    Links to Equations (8) and (11) in the paper:
        - Φ(T_i, E_i, θ): Elastic energy potential returned by the model.
        - S_i = ∂Φ(T_i, E_i, θ)/∂E_i: Stress derived via backpropagation.
        - C = ∂²Φ/∂E²: Stiffness tensor from second derivatives.

    Inputs:
        - n_tiling_params: Number of design parameters.
        - inputs: Combined tensor of tiling parameters T and strain E.
        - model: Neural network representation of Φ(T, E).

    Outputs:
        - psi: Energy density Φ.
        - stress: Voigt representation of the stress tensor S (shape: batch_dim x 3).
        - stiffness tensor: Voigt representation of the stiffness tensor C (shape: batch_dim x 3 x 3).
    """    
    batch_dim = inputs.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(inputs)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            psi = model(inputs, training=False)# Evaluate energy density Φ(T, E).
            dedlambda = tape.gradient(psi, inputs)# Compute stress S = ∂Φ/∂λ, λ = [T, E].
            
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3]) # Extract Voigt stress.

    #C = ∂S/∂E = ∂²Φ/∂E², slicing stress jacobian excluding T        
    C = tape_outer.batch_jacobian(stress, inputs)[:, :, n_tiling_params:]
    del tape
    del tape_outer
    return psi, stress, C

@tf.function
def energyDensity(ti, uniaxial_strain, model):
    """
    Computes:
        1. energy density: Φ(T, E) for given tiling parameters T and uniaxial strain E in voigt form.

    Links to Equation (8) in the paper:
        - Φ(T_i, E_i, θ): Elastic energy potential returned by the model.

    Inputs:
        - ti: Tiling parameters T.
        - uniaxial_strain: Strain tensor E in Voigt form (shape: batch_dim x 3).
        - model: Neural network representation of Φ(T, E).

    Outputs:
        - psi: Energy density Φ.
    """
    batch_dim = uniaxial_strain.shape[0]
    ti = tf.expand_dims(ti, 0)# Expand T to match batch dimensions.
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(uniaxial_strain)# Watch uniaxial strain for differentiation.
        ti_batch = tf.tile(ti, (batch_dim, 1))# Tile T for each batch sample.
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)# Combine T and E as input.
        psi = model(inputs, training=False)# Evaluate energy density Φ(T, E).
    del tape
    return tf.squeeze(psi) # Return energy density.

@tf.function
def objGradUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):
    """
    Computes:
        1. Directional stress magnitutue: d^T S d (scalar strain in fixed theta direction), where S is the stress tensor, and d is a direction vector.
        2. Gradient w.r.t. tiling parameters T: ∂(d^T S d)/∂T.
        3. Gradient w.r.t. strain E: ∂(d^T S d)/∂E.

    Links to Equations:
        - Equation (9a): Outer objective for uniaxial stress design.
        - Equation (9c): Inner optimization constraint, minimized elastic energy Φ orthogonal to direction d.

    Inputs:
        - n_tiling_params: Number of design parameters.
        - ti: Tiling parameters T (shape: n_tiling_params).
        - uniaxial_strain: Strain tensor E in Voigt form (shape: batch_dim x 3).
        - theta: Angle θ specifying the direction vector d.
        - model: Neural network representation of Φ(T, E).

    Outputs:
        - dTSd: The scalar value of d^T S d (shape: batch_dim x 1).
        - grad: Gradient of d^T S d w.r.t. T (shape: batch_dim x n_tiling_params).
        - dOdE: Gradient of d^T S d w.r.t. E (shape: batch_dim x 3).
    """    
    batch_dim = uniaxial_strain.shape[0]
    thetas = tf.tile(theta, (uniaxial_strain.shape[0], 1)) # Repeat angle θ for batch samples.
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1) # Direction vector d (2x1).
    d = tf.cast(d, tf.float64) 
    ti = tf.expand_dims(ti, 0) # Expand T to match batch dimensions.
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ti) # Watch tiling parameters T.
        tape.watch(uniaxial_strain)  # Watch strain E.
        ti_batch = tf.tile(ti, (batch_dim, 1)) 
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1) # Combine T and E as input.
        psi = model(inputs, training=False) # Evaluate energy density Φ(T, E).
        dedlambda = tape.gradient(psi, inputs) # Compute gradient ∂Φ/∂λ, λ = [T, E].
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3]) # Extract Voigt stress.

         # Convert Voigt stress to tensor form for tensor operations.
        stress_xx = tf.gather(stress, [0], axis = 1) 
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

        # Compute stress tensor in direction d, Sd = S d (matrix-vector product).
        Sd = tf.linalg.matvec(stress_tensor, d)
        
        # Compute stress magnidute in direction, d^T S d (inner product).
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
        # print(dTSd)
    grad = tape.jacobian(dTSd, ti) # Gradient of stress magnitute dTSd w.r.t. T.
    dOdE = tape.jacobian(dTSd, uniaxial_strain) # Gradient of stress magnitute dTSd w.r.t. E.
    del tape
    return tf.squeeze(dTSd), tf.squeeze(grad), tf.squeeze(dOdE)  #return strain magnitutute and its gradient w.r.t. T and E

@tf.function
def objUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):

    # same to previous function but only output stress magnite without derivative information
    batch_dim = uniaxial_strain.shape[0]
    thetas = tf.tile(theta, (uniaxial_strain.shape[0], 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    ti = tf.expand_dims(ti, 0)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ti)
        tape.watch(uniaxial_strain)
        ti_batch = tf.tile(ti, (batch_dim, 1))
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
        psi = model(inputs, training=False)
        dedlambda = tape.gradient(psi, inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))

        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.expand_dims(tf.einsum("ij,ij->i",d, Sd), 1)
    del tape
    return tf.squeeze(dTSd)

@tf.function
def objGradStiffness(ti, uniaxial_strain, thetas, model):
    """
    Computes:
        1. Directional stiffness: (d^T C d)^(-1), where C is the stiffness tensor.
        2. Gradient of stiffness w.r.t. tiling parameters T.
        3. Gradient of stiffness w.r.t. strain E.

    Links to Equations:
        - Equation (10a): Objective for directional stiffness design.
        - Equation (11): Stiffness tensor C = ∂²Φ/∂E².

    Inputs:
        - ti: Tiling parameters T (shape: n_tiling_params).
        - uniaxial_strain: Strain tensor E in Voigt form (shape: batch_dim x 3).
        - thetas: Array of angles specifying directions for stiffness calculation.
        - model: Neural network representation of Φ(T, E).

    Outputs:
        - stiffness: Directional stiffness (shape: batch_dim x 1).
        - grad: Gradient of stiffness w.r.t. T (shape: batch_dim x n_tiling_params).
        - dOdE: Gradient of stiffness w.r.t. E (shape: batch_dim x 3).
    """    
    batch_dim = uniaxial_strain.shape[0]
    
    thetas = tf.expand_dims(thetas, axis=1) # Expand angles for batch processing.
    
    # Compute d_voigt for directional stiffness calculation.
    d_voigt = tf.concat((tf.math.cos(thetas) * tf.math.cos(thetas), 
                        tf.math.sin(thetas) * tf.math.sin(thetas), 
                        tf.math.sin(thetas) * tf.math.cos(thetas)), 
                        axis = 1)

    ti = tf.expand_dims(ti, 0)
    with tf.GradientTape(persistent=True) as tape_outer_outer:
        tape_outer_outer.watch(ti)
        tape_outer_outer.watch(uniaxial_strain)
        with tf.GradientTape() as tape_outer:
            tape_outer.watch(ti)
            tape_outer.watch(uniaxial_strain)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(ti)
                tape.watch(uniaxial_strain)
                ti_batch = tf.tile(ti, (batch_dim, 1))
                inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
                psi = model(inputs, training=False)  # Evaluate energy density Φ(T, E).
                stress = tape.gradient(psi, uniaxial_strain) # Compute stress S = ∂Φ/∂E.
        C = tape_outer.batch_jacobian(stress, uniaxial_strain)  # Compute stiffness tensor C = ∂S/∂E.
        
        # Calculate directional stiffness for each direction d_voigt.
        Sd = tf.linalg.matvec(tf.linalg.inv(C[0, :, :]), d_voigt[0, :]) # Solve Sd = C^-1 d.
        dTSd = tf.expand_dims(tf.tensordot(d_voigt[0, :], Sd, 1), axis=0) # Compute scalar d^T C^-1 d.
        #This scalar represents the material's compliance in the given direction  d, or the inverse of stiffness.
        
        stiffness = tf.divide(tf.constant([1.0], dtype=tf.float64), dTSd) # Compute directional stiffness.

        #Repeat for All Directions in the Batch
        for i in range(1, C.shape[0]):
            
            Sd = tf.linalg.matvec(tf.linalg.inv(C[i, :, :]), d_voigt[i, :])
            dTSd = tf.expand_dims(tf.tensordot(d_voigt[i, :], Sd, 1), axis=0)
            stiffness = tf.concat((stiffness, tf.divide(tf.constant([1.0], dtype=tf.float64), dTSd)), 0)
        stiffness = tf.squeeze(stiffness) # Remove unnecessary dimensions.
    grad = tape_outer_outer.jacobian(stiffness, ti) # Gradient of directional stiffness magnitute dTSd w.r.t. T
    dOdE = tape_outer_outer.jacobian(stiffness, uniaxial_strain) # Gradient of directional stiffness magnitute dTSd w.r.t. E
    del tape
    del tape_outer
    del tape_outer_outer
    return tf.squeeze(stiffness), tf.squeeze(grad), tf.squeeze(dOdE)
