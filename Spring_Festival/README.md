# Spring Festival Special: On Simulation-Based Optimization and Learning
## Introduction

To celebrate the Chinese New Year and Spring Festival we will restart our reading session! This time we have two papers with one focus on application of simulation-constrained optimizationand one focus on techniques of constructing datasets for simulation.  
As you already learned, or at least accessed to the necessary knowledge previously, there is no tutorial needed.  
In the meantime, I will provide you with a detailed guide for each paper, to relate the content of the paper back to your research.


You will:

- Systematically understand the use case and application scenarios when you have gradient information of a simulation, that include optimization of design shape, initialization condition, and material parameters.
- Review the different methods of computing simulation gradients (including neural networks, adjoint methods, and auto differentiation).
- Learn how to use compact representations for boundary conditions to build a comprehensive learning database.
- Understand multi-modal regression models that can predict multiple outputs simultaneously.

---

## General Overview

- **Paper 1** – *Differentiable solver for time-dependent deformation problems with contact*  
  - [Project](https://cims.nyu.edu/~zh1476/research/diffipc.html)  
  - Thorough examination of how simulation gradients can drive various applications (shape/material/trajectory optimization), using an adjoint-based framework.

- **Paper 2** – *Computational Design of Cold Bent Glass*  
  - [Paper](https://arxiv.org/pdf/2009.03667)  
  - [Code](https://github.com/russelmann/cold-glass-acm?tab=readme-ov-file)  
  - Demonstrates a differentiable neural surrogate for interactive design of cold bent glass panels, real-time stress estimates, and shape exploration.

---

## Detailed Guide

### Differentiable solver for time-dependent deformation problems with contact

0. **Preview**:
  - This paper presents a general framework enabling diverse applications of simulation-constrained optimization, supporting both static and dynamic problems.
  - It thoroughly compares methodologies and practical examples, making it a valuable reference for thesis development.

1. **Introduction**:
  - Discuss prior research on simulation-constrained optimization and highlight their limitations in achieving generality.
  - Outline the novel features of this work, which enable applications across shape optimization, material property estimation, and trajectory control problems.

2. **Related Work**:
  - Provide a review of differentiable simulators, categorizing them by approach. While this work belongs to the analytic gradient category, it also evaluates neural surrogate approaches.
  - Summarize contact-related simulations and their applications to shape optimization. This section can be skipped unless interested in joint analysis
  - Discuss meshfree methods, emphasizing how remeshing challenges are addressed.

  2.1 ***Choice of Approach to Computing Gradients***:

   - Explain the authors’ motivation for their specific gradient computation approach, aligned with their problem-solving goals. (You may find parallels with your own approach and identify unique requirements.)
   - Highlight the reasoning behind the choice of the adjoint method, though detailed exploration may not be necessary if it diverges from your methods.
   - Note the frequency and impact of discretization in the chosen approach.
   - Comparing discretize-and-optimize vs optimize-and-discretize and how to construct adjoints (AD vs analytics)

3. **Overview**:

  - Introduce the general formulation of the problem, paying close attention to the concatenated parameter functions (q) and their roles.
  - Describe the discretization process and its integration within the overall framework.

4. **Adjoint-Based Objective Derivatives**
  - Contextualize the computation of derivatives using the adjoint approach.
  - Skip detailed derivations and dynamic case analysis unless particularly relevant.
    
5.  **Optimization Algorithm**
   - Provide a high-level overview of the optimization method, which likely aligns with similar frameworks in your work.

6-9. **Details of the Mathematics**
  
  - Focus on specific sections relevant to your work, including:
    - Section 6. Overview of physical parameters optimized in the examples.
    - Section 8. Differentiable categories of force parameters.
    - Section 9. General and specific forms of optimization objectives.
   
10. **Results**
  - This section contains numerous illustrative examples, particularly valuable for reference
    - Section 10.2: Applications on shape optimization, such as the fabricated bridge solution, bridge, 3D beam, interlocking mechanisms, 2D hook, and shock protection.
    - Section 10.4: Applications on material optimization, like sine, bridge, and cube optimizations.
       
### Computational design of cold bent glass façades

**Preview**

  -  This paper tackles a real-world architectural geometry problem: shaping doubly curved glass via cold bending.
  -  The authors build a massive dataset using mechanical simulations of glass panels, then train a multi-modal neural network (Mixture Density Network) to predict both the final shape and maximal stress of each panel, given its boundary.
  -  The advantage: instantaneous feedback in design software, enabling interactive editing and simulation-based (surrogate-based) optimization in real time.
  
**Focus**
  - Section 2, previous works on predicting optimizing the physical performance and machine learning for data-driven design.
  - Section 3, overview of the workflow understanding what is being predicted and what is being optimized and designed.
  - Section 4, bezier-based panel parameterization and compact boundary geometry representations enabling input to a neural network.
  - Section 5, optimization framework, and the motivation of why a neural network will be needed.
  - Section 6, important section about how the model is chosen and trained to predict multiple outputs, and how the dataset is constructed
  - Section 7, practical interactive design tool which uses the neural network. interested to see how they use optimization to 'guide' design
  - Section 8, result section as you can take reference on the scale of their dataset and how they evaluate the prediction model.

### Closing Notes ###

  - Paper 1: A broad, adjoint-based PDE solver that handles force, shape, material differentiation. Good for systematic shape or parameter optimization with guaranteed robustness.
  - Paper 2: Shows how to build and use a surrogate from huge simulation data, giving near-instant design feedback for a real engineering scenario.
  Compare these strategies—direct adjoint vs. neural surrogate—for your own thesis tasks.

  Feel free to skip detailed derivations that do not apply to your own problem.
  Instead, focus on how each paper formulates the problem, handles constraints, structures the optimization, and validates the results.These are key takeaways for applying similar methods in your own thesis work. 
  
  Happy reading! 新年快乐!
