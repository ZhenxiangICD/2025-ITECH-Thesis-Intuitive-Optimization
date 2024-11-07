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

1. **Detail Implementation**
   - Derive the process from
\[
\varphi_i(x) = \frac{1}{2} X^T A_i X + b_i^T X + c_i = 0, \quad i = 1, \cdots, N,
\] to
\[ 
\H x = r
\]
   - What modeling tricks does the author use to make the constraints at most quadratic?
   - What is the interactive modeling procedure with handle-based deformation?

2. **Constraint Terms**
   - Pick a specific constraint (discrete orthogonal or planarity) and write it down in the form of 
   - TBD
   - TBD
     
3. **Fairness and Regularization Terms**
   - TBD
   - TBD
   - TBD
  
4. **Discussion**
   - What does the mesh data structure help to define the constraint-variable relationship?
   - What modeling tricks does the author use to make the constraints at most quadratic?
   - Read the code geometrylab/optimization/guidedprojection.py and geometrylab/optimization/guidedprojectionbase.py carefully and learn how to add iterative constraints to the model.

## Additional Resources

- **Matrix Calculus**
  
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  - [online tool](https://www.matrixcalculus.org/)

- **Video Tutorial 1 (Newton's method in constrained optimization)**

  - [Youtube](https://www.youtube.com/watch?v=7Z1p-cj36_U&ab_channel=KevinTracy)
    
- **Video Tutorial 2 (Levenberg-Marquadt Method (regularized Gauss-Newton), OPTIONAL)**
  
  - [Youtube](https://www.youtube.com/watch?v=2ToL9zUR8ZI&ab_channel=EngineeringEducatorAcademy)

- **Notes**

  - [Gauss-Newton](https://ee263.stanford.edu/lectures/annotated/14_gauss_newton.pdf)
    
- **Quiz** 
  - [Matrix derivatives TBD](https://github.com/ZhenxiangICD/2025-ITECH-Thesis-Intuitive-Optimization)
  - [Newton_Method exercise TBD](https://github.com/ZhenxiangICD/2025-ITECH-Thesis-Intuitive-Optimization)
