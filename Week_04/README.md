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
Most simulation tools' APIs provide this matrix, or at least easy access to assemble it. Computing its inverse might still be expensive, but we can use the adjoint method to solve it with only its transpose. We don't go deep into this this week, but it will be addressed in the future.

*dg/dp* depends more on what parameters you choose. Deriving it by hand is certainly possible and shouldn't be too complex, but automatic differentiation is usually more helpful.

In a neural network, however, the sensitivity matrix *dx(p) / dp* can be easily computed via backpropagation. 

In this week's core paper, you’ll see how neural networks are integrated with FEM knowledge to address inverse design challenges.

We also provide two optional papers for you to split reading, which explore differentiable descriptions of intrinsic relationships between points(geodesic distance and Voronoi diagrams), emphasizing implicit modeling.

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
   - Quickly glance through *FEM Lightweight Intro* slides to familiarize yourself with concepts like strains and energy density. Refer back to them if you encounter difficulties during reading.

---

## While Reading

1. **Overall Aims**:
   - **Core**: Understand how the neural network is trained using simulation data (input-output mappings) and integrated with simulation knowledge for inverse design. Focus on how the optimization process solves nested problems with constraints.
   - **Options**: Learn the concept of implicitly describing a system. Pay attention to the smoothing processes applied during topological transitions.

2. **Details to Skip**:
   - **Options**: Skip most of the equations detailing specific applications (e.g., deformation or dynamics) and focus on the system design.

---

## After Reading

1. **Overall Method**:
   - **Core**: What are the advantages of using a neural network over directly optimizing a simulation? In the inverse design process, why is E required as a variable?
   - **Options**: Explain how the system ensures differentiability.

2. **Optimization**:
   - **Core**: Why is inner optimization necessary, and how is it solved in the paper?
   - **Options**: (Not required this week) Optimization details in the optional papers will be discussed later.

3. **Discussion**:
   - Reflect on different strategies for handling couplings in a system:
     - Boundary coupling in the Voronoi paper
     - Host-object and elastic-curve coupling in the geodesic paper
     - Strain conditions in the core paper

---

## Additional Resources

### Tutorials:
- [Implicit Differentiation](https://www.3blue1brown.com/lessons/implicit-differentiation)
- [Adjoint Sensitivities](https://www.youtube.com/watch?v=MlHKW7Ja-qs&ab_channel=MachineLearning%26Simulation)

### Course and Notes:
- [Sensitivity Analysis](https://crl.ethz.ch/teaching/computational-fab-19/slides/sensitivityAnalysis.pdf)
- [FEM Lightweight Intro](https://www.cs.cmu.edu/~scoros/cs15869-s15/lectures/08-FEM.pdf)

### Code Repository:
- [Wukong: Collection of Differentiable Simulation Problems](https://github.com/liyuesolo/Wukong2024/tree/master)

### Lecture (Optional):
- [Differentiable Simulation](https://www.youtube.com/watch?v=atCFu-vwyVw&t=1261s&ab_channel=%E6%9C%B1%E5%AD%90%E5%8E%9A)
