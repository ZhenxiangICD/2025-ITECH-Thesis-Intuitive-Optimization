# Week 3: Non-linear problems and methods

## Introduction


## Reading Assignments

- **Geometric Optimization via Composite Majorization**
  - [Paper](https://www.geometrie.tuwien.ac.at/geom/ig/publications/2014/formfinding22014/formfinding22014.pdf) 



## Prepare Before Reading 

- Go through slides of *Introduction to Optimization for Simulation* and

  1.) Learn what problems elastic energy can solve in geometry processing;

  2.) Review the general methods of solving a minimization problem

  3.) Understand the three major challenges in geometry optimization and know the names of the methods to address those challenges (don't need to go deep now, you can go back and check whenever you face similar challenges) 

- Watch video tutorial 1 and learn the concept of majorization minimization (MM). It will teach you the basic principles of surrogating a complex function.

- Choose between (hope you can choose differently and discuss with each other at some point!)
  a. ) Watch video tutorial 2 and deeply understand the Laplacian Operator in geometry application (useful to process intrinsic properties). Then read paper 
  b. ) Watch video tutorial 3 and deeply understand the Augmented Lagrangian Method (useful to handle complex constraints). Then read paper 

## Questions

1. **Implementation Details**
   - Derive and understand the process of linearizing a quadratic constraint. 
   - Derive and understand the process from minimizing a sum-of-square problem to solving a linear equation (hint: taking the Gradient and setting it to zero).
   - What modeling tricks does the author use to make the constraints at most quadratic?
   - What is the interactive modeling procedure with handle-based deformation?

2. **Constraint Terms**
   - Pick a specific constraint (e.g., planarity) and write down its form of Hx = r at iteration step n. Make sure to match the dimensions between the matrix and vector.
   - How are inequality constraints modeled?
     
3. **Fairness and Regularization Terms**
   - Why is the fairness term necessary for this optimization method? How to design this term?
   - Understand the purpose of *ϵI* in optimization (Tutorial 2 may help, starting at 25:00) 
  
4. **Discussion**
   - Read the code structure of [guidedprojection.py](https://github.com/WWmore/geometrylab/blob/main/optimization/guidedprojection.py) and [guidedprojectionbase.py](https://github.com/WWmore/geometrylab/blob/main/optimization/guidedprojectionbase.py) and learn how to incrementally add constraints to the model.
   - How does the mesh data structure help to define the constraint-variable relationship?

## Additional Resources

- **Introduction to Optimization for Simulation**
  
  - [slides](https://www.cs.columbia.edu/~honglinchen/assets/docs/teaching/SCA2024_intro_to_optimization.pdf)

- **Video Tutorial 1 (Majorization Minimization)**
  
  - [Youtube](https://www.youtube.com/watch?v=2ToL9zUR8ZI&ab_channel=EngineeringEducatorAcademy)
  - [Bilibili](https://www.bilibili.com/video/BV1Zu4y1x7df?spm_id_from=333.788.videopod.sections&vd_source=2685748f21cc03829a6868afaba6584e)
    
- **Video Tutorial 2 (Augmented Lagrangian Method)**

  - [Youtube](https://www.youtube.com/watch?v=7Z1p-cj36_U&ab_channel=KevinTracy)

- **Video Tutorial 3 (Laplacian Operator)**

  - [Youtube](https://www.youtube.com/watch?v=7Z1p-cj36_U&ab_channel=KevinTracy) 
  
- **Supplementary Video**

  This video (0-45:00) is the PhD thesis dissertation from Emily Whiting, in which she demonstrated the application of gradient-based methods in solving feasibility-based inverse static problems, which is highly relevant to your context.
  She also showed one case of optimization in 'parametric' modeling, in which variables are intrinsic to the design shape, and one case in 'free-form' modeling, in which variables are extrinsic on vertex coordinates.
  I hope this can answer your questions from our last discussion and help you understand the differences between those two approaches.

  - [Youtube](https://www.youtube.com/watch?v=1Qs17UCnZfU&t=2980s&ab_channel=EmilyWhiting)
  - [Paper - Parametric Model]()
  - [Paper - ]
    
