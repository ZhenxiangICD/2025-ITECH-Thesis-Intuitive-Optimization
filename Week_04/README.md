# Week 4: Implicit Function Theorem, Sensitivity Analysis - Intro to Inverse Design
## Introduction

This session introduces the concept of inverse design, a framework for optimizing the equilibrium state of a model under constraints. This type of optimization, often referred to as nested or bi-level optimization, focuses on solving problems where the simulation or optimization outcome depends on input parameters, requiring sensitivity analysis to compute derivatives:

*dx(p) / dp*, 

where *x* is the simulation/optimization outcome derived from parameter *p*.

Let's assume:

*x(p) = argmin E(x_hat, p)*

Here, *x* reflects the configuration at which a function *E* (e.g. deformation energy) is minimized, depending on the parameter set *p* (e.g., external forces, material properties) and the current configuration *x_hat* (e.g., current vertex positions).

Taking the gradient of of *E* w.r.t both *p* and *x* as a function *g*, we define: 

*g(x, p) = 0*

at the minimization point. Using the **implicit function theorem**, we derive the sensitivity:

*dx(p) / dp = - inv(dg/dx) * dg/dp* (Inverse of the Jacobian *dg/dx* multiply the partial derivative of *g* w.r.t. *p*)

In a FEM system, the Jacobian *dg/dx* is essentially the stiffness matrix K! And it's usually easy to express analytically given any simulation problem since most of them are just solving the problem of KΔx = u at every step Δx(nature of PDE). 
Most simulation tools' APIs provide this matrix, or at least easy access to assemble it. Computing its inverse might still be expensive, but we can use the adjoint method to solve it with only its transpose.

*dg/dp* depends more on what parameters you choose. Deriving it by hand is certainly possible and shouldn't be too complex, but automatic differentiation is usually very helpful.

In a neural network, however, the sensitivity matrix *dx(p) / dp* can be easily computed via backpropagation. 

In this week's core paper, you’ll see how neural networks are integrated with FEM knowledge to address inverse design challenges.

We also provide two optional papers for you to split reading, which explore differentiable descriptions of intrinsic relationships between points(geodesic distance and Voronoi diagrams)

---

## Reading Assignments

- **Core**: *Neural Metamaterial Networks for Nonlinear Material Design*
  - [Paper](https://arxiv.org/pdf/2309.10600)
  - [Video](https://www.youtube.com/watch?v=NHLYxoZ2O_s&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/liyuesolo/NeuralMetamaterialNetwork)
 
- **Option A**: *Differentiable Voronoi Diagrams for Simulation of Cell-Based Mechanical Systems*
  - [Paper](https://arxiv.org/pdf/2404.18629)
  - [Short Video](https://www.youtube.com/watch?v=wbBJ4v9VyR0&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/lnumerow-ethz/VoronoiCellSim)
 
- **Option B**: *Differentiable Geodesic Distance for Intrinsic Minimization on Triangle Meshes*
  - [Paper](https://arxiv.org/pdf/2404.18610)
  - [Short Video](https://www.youtube.com/watch?v=R0TByqlbsXQ&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/liyuesolo/DifferentiableGeodesics)

---

## Before Reading

1. **Video Tutorials**:
   - Watch the tutorial on *Implicit Differentiation* to understand when and how derivatives of implicit functions can be computed.
   - Watch the tutorial on *Adjoint Sensitivities* to learn how adjoint methods simplify solving inverse problems.

2. **Slides**:
   - Review the slides on *Sensitivity Analysis* to understand the process of taking derivatives in a simulation.
   - Quickly glance through *FEM Lightweight Intro* slides to familiarize yourself with concepts like strains and energy density. Understand the difference between Cauchy and Green strain. Refer back to them if you encounter difficulties during reading.

---

## While Reading

1. **Overall Aims**:
   - **Core paper**: Understand how the neural network model integrated with simulation knowledge for inverse design. Focus on how the optimization process solves nested problems and where the derivatives come from.
   - **Option papers**: Learn the concept of implicitly describing a system with a governing condition. 

2. **Details to Skip**:
   - **Option papers**: Skip most of the equations detailing specific applications (e.g., deformation or dynamics) and focus on the system modeling.

---

## After Reading

1. **Overall Method**:
   - **Core paper**:
     - What are the advantages of using a neural network over directly optimizing a native-scale simulation?
     - Understand the simulation process to generate data, and how deformation gradient, Cauchy/Green strain and energy dentisy are defined.
   - **Option papers**: Explain the degree of freedom in the implicit system and how they are differentiable.
     
2. **Optimization**:
   - **Core paper**:
     - What is the inner optimization actually defiend? How is it solved and to what derivatives are required? 
     - How are the analytical derivatives of the neural networks applied to the optimization model?
   - **All papers**: Try to take one of the nested problems and derive their derivatives and (optionally) hessian through chain rule.

3. **Discussion**:
   - Reflect on high-level strategy of optimzing a mesh-based simulation with mesh-free representation.

4. **Implementation**:
   - Compare the core paper and its code repository, focusing on the following scripts
     - [Derivatives.py](https://github.com/liyuesolo/NeuralMetamaterialNetwork/blob/main/Projects/NeuralMetamaterial/python/Derivatives.py)
     - [Optimization.py](https://github.com/liyuesolo/NeuralMetamaterialNetwork/blob/main/Projects/NeuralMetamaterial/python/Optimization.py)
     
     - [OptUniaxialStress.py](https://github.com/liyuesolo/NeuralMetamaterialNetwork/blob/main/Projects/NeuralMetamaterial/python/OptUniaxialStress.py)
     - [OptStiffness.py](https://github.com/liyuesolo/NeuralMetamaterialNetwork/blob/main/Projects/NeuralMetamaterial/python/OptStiffness.py)
     - [OptPoissonRatio.py](https://github.com/liyuesolo/NeuralMetamaterialNetwork/blob/main/Projects/NeuralMetamaterial/python/OptPoissonRatio.py)
 
    Understand their correponding parts in the paper. Specifically, you learn

    1.) how to use automatic differentiation in tensorflow to 'watch' the variables to arrive at their derivative
   
    2.) how to use scipy.optimize for an optimization model with Jacobian, Hessian and constraints

   EXTRA: I've added comments to some of the codes to [Annotated_code](https://github.com/ZhenxiangICD/2025-ITECH-Thesis-Intuitive-Optimization/edit/main/Week_04/Annotated_code) for you to better understand what each step is doing. Please pay specific detail to
   
   - *Line 412-446* from Optimization.py to understand the computation of sensitivity analysis
   - *Line 448-496* from OptPoissonRatio.py to understand the global process of an optimization-based optimization 


---

## Additional Resources

### Tutorials:
- [Implicit Differentiation](https://www.3blue1brown.com/lessons/implicit-differentiation)
- [Adjoint Sensitivities](https://www.youtube.com/watch?v=MlHKW7Ja-qs&ab_channel=MachineLearning%26Simulation)

### Course and Notes:
- [Sensitivity Analysis](https://crl.ethz.ch/teaching/computational-fab-19/slides/sensitivityAnalysis.pdf)
- [FEM Lightweight Intro](https://www.cs.cmu.edu/~scoros/cs15869-s15/lectures/08-FEM.pdf)

### Code Repository:
- [Wukong: Collection of Differentiable Simulation Projects](https://github.com/liyuesolo/Wukong2024/tree/master)

### Lecture (Optional):
- [Differentiable Simulation](https://www.youtube.com/watch?v=atCFu-vwyVw&t=1261s&ab_channel=%E6%9C%B1%E5%AD%90%E5%8E%9A)
