# Week 1: Fundamentals of Taylor Series and KKT conditions

## Reading Assignments

- **Interactive Exploration of Design Trade-Offs**
  - [Paper](https://homes.cs.washington.edu/~adriana/tradeoffs/aschulz2018.pdf)
  - [Project](https://homes.cs.washington.edu/~adriana/tradeoffs/index.html)

- **Pareto Gamuts: Exploratory Design for Performance Criteria**
  - [Paper](https://paretogamuts.csail.mit.edu/Pareto_Gamuts_Paper_Final.pdf)
  - [Project](https://paretogamuts.csail.mit.edu/)
  - [Code]()

## Prepare Before Reading

- Review the concept of derivatives and partial derivatives.
- Understand the basics of Taylor Series expansion.
- Understand Lagrange multipliers and the KKT conditions.
- Video and textbook material: see the bottom

## Questions

1. **Paper structure and comparision**
   - Comparing two papers, which aspect does the second one add to the first?
   - How does the mathematical methods in the first paper set up the second?
   - What does a 'context' mean in the second paper? What's its relationship to parameter and constraints?

2. **First-Order Approximation**
   - Derive first- and second- order approximation with a specific [multivariable polynomial example](https://mathinsight.org/taylor_polynomial_multivariable_examples)
   - How does the first-order approximation method determine the directions of movement in the design-context space?
   - How much more computational cost will be for a higher-order approximation and why didn't the author use it?
     
3. **KKT Perturbation**
   - Derive some KKT conditions given a multivariate function with multiple constraints (including equality and inequality)
   - Explain how the KKT conditions are used to form the basis for the first-order approximation in the Pareto exploration.
   - What does the author do to ensure the KKT conditions are not violated during pertubation?
  
4. **Data Structure**
   - What information does a single patch contain?
   - Expalin the performance buffer and what does it store?

## Additional Resources

- **Calculus Tutorial**
  
  - [3Blue1Brown](https://www.3blue1brown.com/topics/calculus)

- **KKT Conditions**

  - [Youtube](https://www.youtube.com/watch?v=uh1Dk68cfWs)
    
- **Adriana Schultz Talk: Robotics for the Next Manufacturing Revolution**
  
  - [Youtube]([https://www.3blue1brown.com/topics/calculus](https://www.youtube.com/watch?v=tYGcGGNZyGc))
  
