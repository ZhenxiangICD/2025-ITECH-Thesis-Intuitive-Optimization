# Week 2: Matrix Calculus, Second-Order Methods (Newton-Raphson, Gauss-Newton)

## Reading Assignments

- **Form-finding with Polyhedral Meshes Made Simple**
  - [Paper](https://www.geometrie.tuwien.ac.at/geom/ig/publications/2014/formfinding22014/formfinding22014.pdf) 
  - [Algorithm Documentation](https://www.huiwang.me/mkdocs-archgeo/optimization/)
  - [Code](https://github.com/WWmore/geometrylab)

## Prepare Before Reading 

- Read Chapter 2 of *The Matrix Cookbook* and grow familiar with taking derivatives of a Matrix expression
- Watch video tutorial 1 and learn the basics of Newton's method and Gauss-Newton method
- Complete the quiz

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

- **Matrix Calculus**
  
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  - [online tool](https://www.matrixcalculus.org/)

- **Video Tutorial 1 (Newton's method in constrained optimization)**

  - [Youtube](https://www.youtube.com/watch?v=7Z1p-cj36_U&ab_channel=KevinTracy)
    
- **Video Tutorial 2 (Levenberg-Marquardt Method)**
  
  - [Youtube](https://www.youtube.com/watch?v=2ToL9zUR8ZI&ab_channel=EngineeringEducatorAcademy)

- **Notes**

  - [Gauss-Newton](https://ee263.stanford.edu/lectures/annotated/14_gauss_newton.pdf)
    
- **Short Quiz** 
  1. What is a Hessian to a scalar function? what is the dimension of its Hessian?
  2. Try to compute the Hessian of a simple sum-of-square function like *(𝑥 [ 𝑖 − 1 ] − 2 ⋅ 𝑥 [ 𝑖 ] + 𝑥 [ 𝑖 + 1 ]) ^ 2 for each i from 1 to len(𝑥) − 1* and understand its sparse nature in such problems
  3. What is a positive semi-definite matrix and what does it imply to an optimization problem?
  4. What is the difference between solving an overdetermined problem and an underdetermined problem with Newton's method?
