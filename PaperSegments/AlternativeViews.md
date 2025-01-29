## Alternative Views

While the hybrid PINN-KAN approach offers advantages in handling noisy PDEs, there do exist various alternative methods and improvements to existing techniques.

- Improving PINN Architectures:
    - Some researchers explore the idea that refining existing PINN architectures might address challenges related to noise.
    - Techniques such as adaptive loss balancing and improved regularization methods are discussed as potential ways to enhance the performance of PINNs in noisy environments.
- Traditional Numerical Solvers:
    - Classical numerical methods, including finite difference and finite element approaches, have a long history of development and optimization.
    - These methods typically operate through a stepwise process, starting from an initial condition and selecting a timestep to estimate solutions iteratively.
    - There is a recognized tradeoff where larger timesteps may lead to increased errors, while smaller timesteps can result in higher computational costs.
- Resilience of Traditional Solvers:
    - Traditional numerical solvers may be adapted for noise handling without the use of deep learning techniques.
    - Techniques such as stochastic finite elements and Monte Carlo methods are mentioned as ways to potentially enhance the robustness of these solvers in uncertain environments.
- Criticism of Hybrid Models:
    - Hybrid models, such as PINN-KAN approaches, may introduce additional complexity and computational demands.
    - There are discussions around whether simpler methods using established solvers or refined PINNs could be adequate for certain PDE problems.
- Alternative Noise-Handling Methods:
    - Various alternative techniques, including stochastic collocation, filtering methods, and Bayesian inference, have been utilized to manage noise in PDEs without relying on neural network solutions.
    - These approaches are noted for their potential interpretability and computational efficiency compared to deep learning models.

Despite these criticisms, PINNs offer a significant advantage over traditional solvers by avoiding the timestep tradeoff. Unlike classical numerical solvers, which require careful selection of timestep sizes to balance accuracy and computational efficiency, PINNs can learn a continuous representation of the solution without being constrained by discrete time-stepping. This makes them particularly effective for solving complex PDEs across varying spatial and temporal scales.


Stuff I read: 
* https://arxiv.org/html/2410.13228v1
* https://arxiv.org/html/2410.00422v1
* https://www.nature.com/articles/s44172-024-00303-3
* https://in.mathworks.com/help/matlab/math/partial-differential-equations.html
* https://www.nature.com/articles/s41598-022-11058-2
* https://arxiv.org/html/2407.19421v1
* https://openreview.net/forum?id=aVlDNbvmCK (check this out guys)
* https://spj.science.org/doi/10.34133/research.0147
* https://ieeexplore.ieee.org/document/10114016
* https://www.computerscijournal.org/vol10no1/noise-removal-and-filtering-techniques-used-in-medical-images/
* https://onlinelibrary.wiley.com/doi/10.1155/2021/8176746 (weiner filter???)
* https://ieeexplore.ieee.org/document/9991317
* https://www.tandfonline.com/doi/full/10.1080/17499518.2024.2315301#abstract
* https://openreview.net/forum?id=z9SIj-IM7tn
* https://www.researchgate.net/publication/360438101_Self-adaptive_loss_balanced_Physics-informed_neural_networks
* https://arxiv.org/abs/2211.15498v4 (pinn and noise)
* https://www.aimspress.com/aimspress-data/mbe/2022/12/PDF/mbe-19-12-601.pdf
* https://arxiv.org/html/2211.15498v4 (more noise)
