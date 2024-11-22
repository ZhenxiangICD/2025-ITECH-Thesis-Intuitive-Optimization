# Week 4: Implicit Function Theorem, Sensitivity Analysis - Intro to Inverse Design
## Introduction

This session starts a new discussion on inverse design, that is, to discuss a set of problems that optimizes an equilibrium state of some model of optimization/simulation. This kind of optimization is also known as nested/bi-level optimization.
The key to such a problem lies in computing the derivative of simulation, a.k.a. sensitivity analysis:

*dx(p) / dp*, 

where *x* is the simulation/optimization outcome from parameter *p*.

Let's assume 

*x(p) = argmin E(x_hat, p)*

meaning *x* is the configuration when some function *E* (i.e. deformation energy) reaches minimal, which depends on a parameter set *p* (i.e. external force, material property) and current configuration *x_hat* (i.e. vertex positions).

Writing gradient of *E* w.r.t both *p* and *x* as a function *g*, we achieve: 

*g(x, p) = 0*

at the minimization point. Then comes the rescue of the implicit function theorem, which you will see in more detail in the tutorial. But essentially it tells you that you can have the following equation

*dx(p) / dp = - inv(dg/dx) * dg/dp* (Inverse of the Jacobian *dg/dx* multiply the partial derivative of *g* w.r.t. *p*)

In a FEM system, the Jacobian *dg/dx* is essentially the stiffness matrix K! And it's usually easy to express analytically given any simulation problem since most of them are just solving the problem of KΔx = u at every step Δx(nature of PDE). 
Most simulation tools' APIs provide this matrix, or at least easy access to assemble it. Computing its inverse might still be expensive, but we can use the adjoint method to solve it with only its transpose. We don't go deep into this this week, but it will be addressed in the future.

*dg/dp* depends more on what parameters you choose. Deriving it by hand is certainly possible and shouldn't be too complex, but automatic differentiation is usually more helpful.

In a neural network, however, the sensitivity matrix *dx(p) / dp* can be easily computed via backpropagation. Then you can use the chain rule to optimize a problem involving the prediction from the model.

In the core paper we will read this week, the author also uses a neural network to predict the material behavior... There you can see how he COMBINES the knowledge from FEM and NN to achieve the inverse design of a complex system!

I further provide two more optional papers, like last time you can distribute the tasks. They share a very similar research scope, that is to implicitly describe the intrinsic relationship between different points (geodesic distance and voronoi diagram) in a differentiable manner. 

## Reading Assignments

- **Core - Neural Metamaterial Networks for Nonlinear Material Design**
  - [Paper](https://arxiv.org/pdf/2309.10600)
  - [Video](https://www.youtube.com/watch?v=NHLYxoZ2O_s&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/liyuesolo/NeuralMetamaterialNetwork)
 
- **Option A- Differentiable Voronoi Diagrams for Simulation of Cell-Based Mechanical Systems**
  - [Paper](https://arxiv.org/pdf/2404.18629)
  - [Short Video](https://www.youtube.com/watch?v=wbBJ4v9VyR0&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/lnumerow-ethz/VoronoiCellSim)
 
- **Option B- Differentiable Geodesic Distance for Intrinsic Minimization on Triangle Meshes**
  - [Paper](https://arxiv.org/pdf/2404.18610)
  - [Short Video](https://www.youtube.com/watch?v=R0TByqlbsXQ&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/liyuesolo/DifferentiableGeodesics)
    
## Before Reading 

- Watch the video tutorial on *Implicit Differentiation* and understand at which condition the derivative of an implicit function can be computed in the abovementioned manner.

- Read slides of *sensitivity analysis* and review the method of taking derivatives of a simulation process. Then watch the video tutorial on *Adjoint Sensitivities*

- Quickly take a glance through the slides of *FEM lightweight intro* and have the basic concepts of strains and energy density. Go back to them when you have difficulty reading the paper.


## While Reading
1. **Overall Aims**
   - *Core*: Understand how the author trains the model with simulation data (map in and map out) and how the model is used together with simulation knowledge for inverse design. The optimization part is crucial as it reveals the key to solving a nested problem with constraints.
   - *Option*: Try to grab the concept of describing a system implicitly. Understand the smoothing process they take during topological transitions.

2. **Details To Skip**
   - For the core paper, check on the FEM tutorial back and forth for strain definitions. Pay attention to how the degree of freedom gets simplified with a full FEM system.
   - For the option paper, skip most of the equations on the application part of the system (i.e. deformation and dynamics) and focus more on the system's design.
    
## After Reading

1. **Overall Method**
   - For the core paper, what are the advantages of using a neural network instead of directly optimizing over a simulation? What are the inputs used to train the network and why are they used?
   - For the option papers, explain how the system is differentiable.
     
2. **Optimization**
   - For the core paper, why is the inner optimization necessary? How is it solved?
   - Optimization part in the option papers is not required at this time...
  
3. **Discussion**
   - Reflect on the different strategy of handing couplings in a system. (Boundary Coupling in the voronoi paper, hosting object - elastic curve coupling in the geodesic paper, and the strain condition for the core paper)

## Additional Resources

- **Tutorials**
  - [Implicit Differentiation](https://www.3blue1brown.com/lessons/implicit-differentiation)
  - [Adjoint Sensitivities](https://www.youtube.com/watch?v=MlHKW7Ja-qs&ab_channel=MachineLearning%26Simulation)
    
- **Course and Notes**
  - [Sensitivity Analysis](https://crl.ethz.ch/teaching/computational-fab-19/slides/sensitivityAnalysis.pdf)
  - [FEM lightweight intro](https://www.cs.cmu.edu/~scoros/cs15869-s15/lectures/08-FEM.pdf)

- **Code Repository**
  - [Wukong / collection of many classic differentiable simulation problems)](https://github.com/liyuesolo/Wukong2024/tree/master)
 
- **Lecture (only watch when you have spare time...)**
  - [Differentiable Simulation](https://www.youtube.com/watch?v=atCFu-vwyVw&t=1261s&ab_channel=%E6%9C%B1%E5%AD%90%E5%8E%9A)  

