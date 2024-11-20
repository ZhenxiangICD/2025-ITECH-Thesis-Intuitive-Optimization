# Week 4: Implicit Function Theorem, Sensitivity Analysis - Intro to Inverse Design
## Introduction

This session starts a new discussion on inverse design, that is, to discuss a set of problems that optimizes an equilibrium state of some model of optimization/simulation. This kind of optimization is also known as nested/bi-level optimization.
The key to such a problem lies in computing the derivative of simulation, a.k.a. sensitivity matrix :

*dx(p) / dp*, 

where *x* is the simulation/optimization outcome from parameter *p*.

Let's have 

*x(p) = argmin E(x_hat, p)*

meaning *x* is the configuration when some function *E* (i.e. deformation energy) reaches minimal, which depends on a parameter set *p* (i.e. external force, material property) and current configuration *x_hat* (i.e. vertex positions).

Writing gradient of *E* w.r.t both *p* and *x* as a function *g*, we achieve: 

*g(x, p) = 0*

at the minimization point. Then comes the rescue of the implicit function theorem, which you will see in more detail in the tutorial. But essentially it tells you that you can have the following equation

*dx(p) / dp = - inv(dg/dx) * dg/dp* (Inverse of the Jacobian *dg/dx* multiply the partial derivative of *g* w.r.t. *p*)

In a FEM system, the Jacobian *dg/dx* is essentially the stiffness matrix K! And it's usually easy to express analytically given any simulation problem since most of them are just solving the problem of Kx = u at every step. 
However, computing its inverse might still be expensive, but we can use the adjoint method to solve it with only its transpose. We don't go deep into this for this week, but it will be addressed in the future.

In a neural network, however, the sensitivity matrix *dx(p) / dp* can be easily computed via backpropagation. Then you can use the chain rule to optimize a problem involving the prediction from the model.

In the paper we will read this week, the author also uses a neural network to predict the material behavior... There you can see how he COMBINES the knowledge from FEM and NN to achieve inverse design of a complex system!

## Reading Assignments

- **Core - Neural Metamaterial Networks for Nonlinear Material Design**
  - [Paper](https://arxiv.org/pdf/2309.10600)
  - [Video](https://www.youtube.com/watch?v=NHLYxoZ2O_s&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/liyuesolo/NeuralMetamaterialNetwork)
 
- **Bonus A- Differentiable Voronoi Diagrams for Simulation of Cell-Based Mechanical Systems**
  - [Paper](https://roipo.github.io/publication/poranne-2013-interactive/planarization.pdf)
  - [Video](https://www.youtube.com/watch?v=wbBJ4v9VyR0&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/lnumerow-ethz/VoronoiCellSim)
 
- **Bonus B- Differentiable Geodesic Distance for Intrinsic Minimization on Triangle Meshes**
  - [Paper](https://arxiv.org/pdf/2404.18610)
  - [Video](https://www.youtube.com/watch?v=R0TByqlbsXQ&ab_channel=ComputationalRoboticsLab)
  - [Code](https://github.com/liyuesolo/DifferentiableGeodesics)
    
## Before Reading 

- Go through slides of *Introduction to Optimization for Simulation* and

  1.) Review the general methods of solving a minimization problem

  2.) Familiarize yourself with three major challenges in geometry optimization and take note of some common methods (no need to dive deep now—you can refer back when encountering similar challenges).

- Watch ONE of the video tutorials on the following algorithms and learn its key concepts. 

  a. ) Augmented Lagrangian Method (useful in managing feasibilities)

  b. ) Alternating Least-Square (useful in parallel and distributed computing)

  Choose paper 1A or 1B based on the algorithm you study (coordinate with peers to choose different ones when possible).

- Watch video tutorial 3 and learn the concept of majorization minimization (MM). This will cover the basic principles of creating a surrogate for a complex function. Then, briefly review the *Mappings* slides — focus on understanding the role of the Jacobian in a mapping and how it can be decomposed, without worrying about all technical details.

## While Reading
1. **Overall Aims**
   - *Paper 1A / 1B* : Compare the methods used in these papers to the Gauss-Newton method we studied last week and examine why the selected methods are applied, especially in relation to constraint definition.
   - *Paper 2* :  Understand the concept of approximating an objective function with convex-concave decomposition and how to derive a majorizer.

2. **Details To Skip**
   - For paper 1B, you can ignore the mathematical details in Section 6, as we will discuss them further in the next reading.
   - For paper 2, warning that there are a lot of abstract formula and derivations! If you are already exhausted after the first reading, please just watch the video and focus on the details and derivation from 7:00 to 10:00. 
    
## After Reading

1. **Method**
   - For paper 1A/1B, discuss the methods you learned from the tutorial and how it is applied in the paper. For Augmented Lagrangian, how does it balance hard and soft constraint satisfaction? For Alternating Least-Square, what is the benefit in using a local/local scheme compared to a local/global scheme?
   - For paper 2, explain how to use the composite nature of geometric objective to build majorizer and Hessian.
     
2. **Interactive Optimization**
   - For paper 1A/1B, discuss what application scenario the fast optimization allow designers to make adjustment to their design. How do these methods enable a flexible design workflow in interactive applications? What is the balance between optimization and user control?
  
3. **Discussion**
   - Discuss how the Jacobian (deformation gradient) captures the intrinsic of the shape change of a triangle. 

## Additional Resources

- **Course and Notes**
  - [Sensitivity Analysis](https://crl.ethz.ch/teaching/computational-fab-19/tutorials/tutorial_a3.pdf)
  - 
  - [FEM lightweight intro](https://www.cs.cmu.edu/~scoros/cs15869-s15/lectures/08-FEM.pdf)

- **Code Repository**
  - [Wukong / collection of many classic differentiable simulation problems)](https://github.com/liyuesolo/Wukong2024/tree/master)

- **Video Tutorial 1 (Augmented Lagrangian Method / ALM)**

    *Learn the concept of nesting an optimization problem and updating penalty weights iteratively to satisfy feasibility conditions.*

  - [Youtube](https://www.youtube.com/watch?v=jyq7_GoT0H4&t=2s&ab_channel=KevinTracy)

- **Video Tutorial 2 (Alternating Least-Square / ALS)**

    *Learn the concept of separating variables and alternatingly optimizing them.*

  - [Youtube1 / least-square basic](https://www.youtube.com/watch?v=8mAZYv5wIcE)
  - [Youtube2 / recommendation system](https://www.youtube.com/watch?v=5im_ZSOZdxI)

- **Video Tutorial 3 (Majorization Minimization)**

     *Learn to create a surrogate objective based on the concept of majorization. Watch the first 10 minutes for the basic concept and go through 1-2 examples to practice.*
  
  - [Youtube](https://www.youtube.com/watch?v=S_QSbmBupLc&ab_channel=ComputationalGenomicsSummerInstituteCGSI)
  - [Bilibili](https://www.bilibili.com/video/BV1Zu4y1x7df?spm_id_from=333.788.videopod.sections&vd_source=2685748f21cc03829a6868afaba6584e)
