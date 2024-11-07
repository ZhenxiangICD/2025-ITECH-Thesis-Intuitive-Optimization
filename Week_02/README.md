# Week 2: Matrix Calculus, Second-Order Methods (Newton-Raphson, Gauss-Newton)

## Reading Assignments

- **Form-finding with Polyhedral Meshes Made Simple**
  - [Paper](https://www.geometrie.tuwien.ac.at/geom/ig/publications/2014/formfinding22014/formfinding22014.pdf) 
  - [Algorithm Documentation](https://www.huiwang.me/mkdocs-archgeo/optimization/)
  - [Code](https://github.com/WWmore/geometrylab)

## Prepare Before Reading 

- Read Chapter 2 Of *The Matrix Cookbook* and grow familiar with taking derivatives of a Matrix expression
- Watch video tutorial 1 and learn the basics of Newton's method and Gauss-Newton method
- Complete the quiz
- Video and textbook material: see the bottom

## Questions

1. **Implementation Details**
   - Derive and understand the process of linearizing a quadratic constraint. 
   - Derive and understand the process from minimizing a sum-of-square problem to solving a linear equation (hint: taking the Gradient and setting it to zero).
   - What modeling tricks does the author use to make the constraints at most quadratic?
   - What is the interactive modeling procedure with handle-based deformation?

2. **Constraint Terms**
   - Pick a specific constraint (e.g., planarity) and write down its form of Hx = r at iteration step n, make sure to match the dimensions between matrix and vector
   - How are inequality constraints modeled?
     
3. **Fairness and Regularization Terms**
   - Why is the fairness term necessary for this optimization method? How to design this term?
   - Understand the purpose of *ϵI* in optimization (Tutorial 2 may help, starting at 25:00) 
  
4. **Discussion**
   - Read the code geometrylab/optimization/guidedprojection.py and geometrylab/optimization/guidedprojectionbase.py carefully and learn how to add iterative constraints to the model.
   - What does the mesh data structure help to define the constraint-variable relationship?

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
    
- **Quiz** 
  - [Matrix derivatives TBD](https://github.com/ZhenxiangICD/2025-ITECH-Thesis-Intuitive-Optimization)
  - [Newton_Method exercise TBD](https://github.com/ZhenxiangICD/2025-ITECH-Thesis-Intuitive-Optimization)
