# Week 5: Laplacian, Spectral Analysis - Shape Intrinsics
## Introduction

After an intensive paper reading week, we will turn back to learn basic mathematic and geometric concepts, with a focus on 'intrinsic' properties of a shape. 
You're expected to spend most of time in those three video tutorials. 
From the lectures you should learn how a general mathematic definition can be turned into specific applications in geometric processing tasks.

This is a review- and summary- ish paper. The expectation is to review what you learn from the videos and see how Laplacian and spectrum analysis are used for feature extraction and shape editing.

---

## Reading Assignment
 
- *Differential Representations for Mesh Processing*
  - [Paper](https://igl.ethz.ch/projects/Laplacian-mesh-processing/STAR/CGF-Laplacian-mesh-processing.pdf)

## Before Reading 

1. **Math Foundation**:
   - Watch the tutorial on *Implicit Differentiation* to understand when and how derivatives of implicit functions can be computed.
   - Watch the tutorial on *Adjoint Sensitivities* to learn how adjoint methods simplify solving inverse problems.

2. **Laplacian**:
   - Watch video tutorial on *Sensitivity Analysis* to understand the process of taking derivatives in a simulation.

2. **Application**:
   - Review the slides on *Sensitivity Analysis* to understand the process of taking derivatives in a simulation.
   - Quickly glance through *FEM Lightweight Intro* slides to
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

### Mathematic Fundamentals:
- [Eigenvalues and eigenfunctions](https://www.3blue1brown.com/lessons/implicit-differentiation)
- [Spectrum Analysis](https://www.youtube.com/watch?v=MlHKW7Ja-qs&ab_channel=MachineLearning%26Simulation)

### Geometry Courses:
- [The Laplace Operator](https://www.youtube.com/watch?v=oEq9ROl9Umk&t=3638s&ab_channel=KeenanCrane)
- [PDE and Spectral Approaches to Geometry Processing](https://www.youtube.com/watch?v=BTZKa0wTfaQ&ab_channel=JustinSolomon)

### Lecture (Optional):
- [Differentiable Simulation](https://www.youtube.com/watch?v=atCFu-vwyVw&t=1261s&ab_channel=%E6%9C%B1%E5%AD%90%E5%8E%9A)
