# Week 3: Non-linear problems / Mappings
## Introduction

This session builds on our previous discussion on optimization by shifting focus from quadratic to non-linear problems. In papers 1A and 1B, you will explore two different solutions to the same problem covered in last week’s readings, with a focus on addressing non-linear constraints. In paper 2, we approach a new type of geometric problem: mapping, which discusses how to transform an original shape into a desired domain while preserving its core properties.


## Reading Assignments

- **1A - Interactive Design Exploration for Constrained Meshes (Augmented Lagrangian)**
  - [Paper](http://www.bdeng.me/DesignExploration_CAD.pdf) 
 
- **1B - Interactive Planarization and Optimization of 3D Meshes (Alternating Least-Square)**
  - [Paper](https://roipo.github.io/publication/poranne-2013-interactive/planarization.pdf) 

- **2 - Geometric Optimization via Composite Majorization**
  - [Paper](https://roipo.github.io/publication/shtengel-2017-geometric/CompMajor.pdf) 
  - [Video](https://dl.acm.org/doi/10.1145/3072959.3073618)
  - [Code](https://github.com/Roipo/CompMajor)
    
## Before Reading 

- Go through slides of *Introduction to Optimization for Simulation* and

  1.) Review the general methods of solving a minimization problem

  2.) Familiarize yourself with three major challenges in geometry optimization and take note of some common methods (no need to dive deep now—you can refer back when encountering similar challenges).

- Watch ONE of the video tutorials on the following algorithms and learn its key concepts. 

  a. ) Augmented Lagrangian Method (useful in managing feasibilities)

  b. ) Alternating Least-Square (useful in parallel and distributed computing)

  Choose paper 1A or 1B based on the algorithm you study (coordinate with peers to choose different ones when possible).

- Watch video tutorial 3 and learn the concept of majorization minimization (MM). This will cover the basic principles of creating a surrogate for a complex function. Then, briefly review the Mappings slides — focus on understanding the role of the Jacobian in a mapping and how it can be decomposed, without worrying about all technical details.

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
   - Discuss how the Jacobian (deformation gradient) captures the intrinsic of a shape change of a triangle. 

## Additional Resources

- **Course and Notes**
  - [Introduction to Optimization for Simulation](https://www.cs.columbia.edu/~honglinchen/assets/docs/teaching/SCA2024_intro_to_optimization.pdf)
  - [Mappings](https://crl.ethz.ch/teaching/shape-modeling-18/lectures/05_Mappings.pdf)
    
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
