# Week 5: Laplacian, Spectral Analysis - Shape Intrinsics
## Introduction

This week, we shift focus to foundational mathematical and geometric concepts, exploring how shapes can be described and manipulated using their intrinsic properties.

The emphasis is on the Laplacian operator and its role in applications like shape editing and geometry processing. These intrinsic descriptions of shapes are powerful tools for tasks such as interpolating, deformation, and mapping.

You will:

- Watch video tutorials to build an understanding of eigenvalues, vector spaces, and PDEs.
- Explore the Laplace operator and its applications in geometry processing.
- Read two papers: the first provides a broad overview of shape editing techniques, while the second addresses an overview with specific focus on using Laplacian representations.
- To balance the reading load, you can alternate between papers and tutorials.

---

## Reading Assignment

- **Paper 1** - *A Revisit of Shape Editing Techniques: from the Geometric to the Neural Viewpoint*
  - [Paper](https://arxiv.org/pdf/2103.01694)
  - Required section: 2.1.1 , 2.2 , 3.1 ; Other sections are optional
  
- **Paper 2** - *Differential Representations for Mesh Processing*
  - [Paper](https://igl.ethz.ch/projects/Laplacian-mesh-processing/STAR/CGF-Laplacian-mesh-processing.pdf)
  - This is essentially a more in-depth introduction to section 2.1.1 of paper 1.


## Before Reading 

1. **Math Fundamentals**:
   - Watch the tutorial videos on *eigenvalues*, *vector spaces*, and *PDEs*.

2. **Laplacian**:
   - Watch the video tutorial on *The Laplace Operator*. You can skip the section on exterior calculus (27:34 - 30:07).

   This tutorial provides an in-depth introduction to the Laplace Operator, often referred to as the "Swiss-Knife" in geometry analysis. After completing the tutorial, start reading the first paper, which is more general and requires fewer technical prerequisites.

3. **Spectral Geometry Processing**:
   - Watch the video tutorial on *PDE and Spectral Approaches to Geometry Processing*.

   This tutorial introduces spectral and PDE-based analysis from an application perspective, including the Laplacian operator’s role in geometry processing. It will strengthen your understanding of these approaches. Afterward, proceed to read Paper 2.

---

## Aims and Questions

Review these questions before watching the tutorials and reading the papers. Answer them after completing both.

### 1. **General Concept Understanding**

#### **The Laplace Operator**:
   - What does the Laplace Operator intuitively describe?  
   - On what objects does a discrete Laplace Operator operate? What is its relationship with the Hessian matrix?  
   - What does Dirichlet energy describe? How is it related to the Laplacian?  
   - Explain different boundary condition scenarios.

#### **Spectral Analysis**:
   - Try *Robust Laplacian* from the provided code repository. Input a list of points from Rhino/Grasshopper (I provided a function to convert a point list to .ply format), which could be as sparse as column grids, or as dense as FEM points across the entire slab domain, use the provided code to test what different scalar outputs corresponding to different eigenvectors look like and summarize what you have observed.
   - What shape features are described by using higher/lower eigenvalues?
   - How does solving a Poisson equation potentially lead to solving an eigenvalue problem?  

#### **Differential Coordinates**:
   - What are the advantages of using differential coordinates?  
   - What is the relationship between Laplacian coordinates and the Laplacian operator?  
   - How can rotation-invariant coordinates and representations be created?  
   - Summarize the methods for constructing differential coordinate representations from an optimization perspective.

---

### 2. **Applications**

#### **Deformation**:
   - Explain how the Laplacian is used in shape editing and why it produces plausible results.  
   - Understand the As-Rigid-As-Possible (ARAP) energy and explain its importance in shape deformation.

#### **Semantic Constraints**:
   - Explain how curve networks can serve as structurally-aware constraints.  
   - What are the common constraints in architectural models, and what methods are used to address them?

#### **Skeletons**:
   - For deformations based on linear blend skinning (LBS), it is crucial to design the linear blend weights (**W**) to control how one point influences the shape. Explain how **W** is constructed in the bounded biharmonic weights (BBWs) method. What properties result from this weight construction?

---

### 3. **Discussion**

   - Think of an inverse deformation problem: if the shape remains unchanged, what criteria would you design to guide users in moving the handle points so that those handles can produce better or similar global control over the shape?
   - Think of heat equation in the context of column-slab. What could the outcome scalar field mean if we assign a value at each column point as the initial boundary condition and solve a heat equation? What if we set a 
   - Select examples listed in the sections related to FEM simulation-aware shape editing and save them for future reference.


---

## Additional Resources

### Mathematic Fundamentals:
- [Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)
- [Vector Spaces](https://www.3blue1brown.com/lessons/abstract-vector-spaces)
- [PDEs](https://www.3blue1brown.com/lessons/pdes)

### Geometry Tutorials:
- [The Laplace Operator](https://youtu.be/oEq9ROl9Umk?si=XN7urauKAiuPbMlM)
- [PDE and Spectral Approaches to Geometry Processing](https://www.youtube.com/watch?v=BTZKa0wTfaQ&ab_channel=JustinSolomon)

### Code Repository
- [Robust Laplacian](https://github.com/nmwsharp/robust-laplacians-py)
