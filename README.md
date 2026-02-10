# Description
This project implements a Denoising Diffusion Probabilistic Model (DDPM) designed to perform discrete arithmetic reasoning within a $2 \times 10$ pixel constraint. By representing numbers and operators as $2 \times 2$ binary glyphs, the "LogicDiffuser" architecture combines a CNN for local spatial feature extraction with a high-capacity MLP bottleneck to bridge the gap between generative image patterns and formal algebraic logic. The system treats mathematical calculation as a conditioned inpainting task: the input equation is held as a fixed boundary condition, while the $2 \times 4$ result region is iteratively denoised from Gaussian noise to its logically consistent state, utilizing coordinate embeddings to distinguish between operand and result segments.

# Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Ari-Hiroko/2-2-diffusion-calculator.git
cd 2-2-diffusion-calculator
pip install -r requirements.txt

```

# Usage

## 1. Running the Complete Script

The script performs training and runs predefined test cases (e.g., , ) automatically:

```bash
python 4PX.PY

```

## 2. Manual Inference

If you want to solve a specific equation using the trained model, you can manually call the `visual_solve` function:

```python
# Example: Calculate 6 % 4
# Arguments: (model, operand_a, operator, operand_b)
result_grid = visual_solve(model, 6, '%', 4)

# Visualize the 2x10 pixel result
plot_solution(result_grid, title="6 % 4 = ?")

```

# Motivation
This project was built purely for fun. There is a certain poetic absurdity in using a sophisticated, multi-step Denoising Diffusion Probabilistic Model just to calculate basic decimal arithmetic that a 1970s pocket calculator could handle instantly. It is an intentionally "over-engineered" experiment—using a complex generative process to output a tiny $2 \times 10$ pixel result is objectively inefficient, which is precisely what makes it a compelling "weird inspiration."
# Drawbacks
Lacks a manual input interface, but I'm too lazy to fix it. 
# Troubleshooting
If the results are incorrect, increase the epochs until you are satisfied. 
# Screenshots
<img width="811" height="1198" alt="M(Y6{ M}FEJ%%22`R0HMKPU" src="https://github.com/user-attachments/assets/acaf451b-1b18-410f-bffb-8300bd116cb7" />
