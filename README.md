An in depth exploration of PINNs and their capabilities when working with unkown shapes and models.

PINNs are very good at calculating simulations for the shapes and parameters they were trained on. In our case this is seen through fluid dynamics PINNs that are trained on airfoils, being good at computing the fluid dynamics of air around airfoils. 
Our goal was to analyze the effect of training a model on multiple different shapes, on their ability to compute the fluid dynamics of a shape they'd never seen before.
Due to lack of testing time and resources, our findings were inconclusive, but could very well serve as a jumping off point for anyone looking to do similar research.

For a more in depth analysis of our results and description of the process, see Final PINN report.pdf

To see the code we used and models that we created, please see the contents of the pinnTorch folder.
