## KAN Implementation
PINNs, despite their potential, struggle with noisy data, often converging to inccorect solutions. In our experiments, we generated noisy data BOBBOBBOB and trained a PINN to approximate the solutions of Burgers' and Helmholtz equations <MORE TO BE ADDED>. However, the PINN failed to generalize under noisy conditions, leading to unstable solutions and poor convergence. To overcome this, we propose a hybrid framework that integrates KANs with PINNs.

KANs, inspired by Kolmogorov’s Superposition Theorem, excel in approximating high-dimensional functions by decomposing them into simpler, one-dimensional mappings. This property makes KANs particularly effective for denoising the outputs of PINNs. Instead of feeding raw noisy data directly into the PINN, we first pass it through a trained KAN, which learns to reconstruct the clean underlying solution from the corrupted observations. This preprocessing step significantly improves the PINN’s stability and accuracy <convergence too??>

Our implementation of KAN follows a fully connected architecture with learnable activation functions that enable adaptive function approximation- a key component to modelling the noise our data has. 
The model is structured as follows: 
- **Input Layer**: 201 neurons <corresponding to the pde or what??? Why 201>
- **Hidden Layers:**  
  - Two layers with **512 neurons** each.  
  - One **1024-neuron bottleneck layer**, capturing complex transformations.  
  - Two additional **512-neuron layers** for feature refinement.  
- **Output Layer:** 201 neurons, producing a denoised version of the input.  

KANs employ learnable activation functions, allowing them to flexibly adapt to the structure of the noise and the underlying PDE dynamics. 

### Training process: <I used GPT for this part, tell if anything is incorrect>
1. Dataset Preparation:  
   - Noisy and clean solutions are stored in an **HDF5 file** and loaded into memory.  
   - The data is reshaped and formatted as **tensor pairs**: noisy input and clean target.  

2. Loss Function & Optimization:
   - The **Mean Squared Error (MSE)** is used to measure the discrepancy between the KAN’s output and the clean target.  
   - The model is optimized using the **Adam optimizer** with an initial learning rate of **1e-3**.  
   - A **ReduceLROnPlateau scheduler** is incorporated to dynamically adjust the learning rate based on validation performance.  

3. Training Strategy:
   - The model is trained in batches of **3072 samples** for efficient optimization.  
   - After each epoch, the learning rate is adjusted if necessary to prevent stagnation.  

4. Evaluation & Deployment:
   - The trained KAN is applied to **denoise unseen noisy PDE solutions**.  
   - Performance is assessed by comparing the MSE between the **KAN-predicted clean solution** and the actual clean solution.  
   - The final output is visualized to ensure qualitative improvements in solution accuracy.  


## PINN-KAN Implementation
To validate our approach, we implemented a KAN model to learn the mapping from noisy to clean data before feeding it into the PINN. Our pipeline consists of the following steps:
<need help here>
1. Data Generation: We simulated solutions to Burgers' and Helmholtz equations and introduced different types of noise BOBBBOBBOB
2. Training a Baseline PINN: The PINN was trained directly on noisy data, and its performance was evaluated. As expected, the model struggled to converge and produced highly inaccurate solutions <should I add clean data pinn too???>
3. Training the KAN for Denoising: We trained a KAN model to learn the transformation from noisy observations to clean solutions
4. Integration with PINNs: <shusrith help>

## Results????
Our experiments demonstrate that the PINN-KAN hybrid framework effectively mitigates the impact of noise in PDE-solving tasks. Compared to the baseline PINN, the hybrid model:
* Achieves better generalization with high noise elevels
* Produces solutions that are more stable and consistent witht he governing equations
By leveraging KANs as a preprocessing step, we bridge the gap between real-world noisy data and the ideal conditions required for PINN training. This hybrid methodology not only improves PDE solvers but also sets a precedent for combining physics-based models with deep learning architectures to enhance robustness in computational physics.

