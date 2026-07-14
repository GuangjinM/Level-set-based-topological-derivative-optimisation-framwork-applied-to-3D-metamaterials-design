# Level-set-based Topological Derivative Optimisation Framework for 3D Metamaterials Design

This repository implements a 3D level-set based topology optimisation framework for metamaterial design. The overall idea is to combine:

- a periodic homogenisation procedure to compute the effective stiffness tensor of a unit-cell microstructure;
- a material objective defined from the homogenised stiffness, which drives the design toward a target mechanical response;
- a topological-derivative / level-set update scheme that iteratively modifies the material distribution.

The implementation is organised around three core files:

- [main_3d.py](main_3d.py): the main optimisation loop and the outer update logic;
- [definition_3d.py](definition_3d.py): PDE definitions, homogenisation, material modelling, and the cost function / gradient;
- [init_3d.py](init_3d.py): numerical parameters, optimisation settings, and material / solver configuration.

## 1. Overall algorithm framework

The workflow can be summarised as follows:

1. Initialise a 3D periodic unit cell and a level-set field.
2. Convert the level-set field into a two-phase material distribution (solid / void).
3. Solve periodic cell problems for several macroscopic strain loading cases to obtain the homogenised stiffness tensor $C_{\mathrm{hom}}$.
4. Evaluate a scalar cost function from $C_{\mathrm{hom}}$ and compute its gradient with respect to the stiffness tensor.
5. Use the chain-rule-based topological derivative to obtain a sensitivity field on the design domain.
6. Update the level-set field through a level-set / line-search based optimisation step.
7. Repeat until convergence, optionally with mesh refinement and volume-control continuation.

In other words, the method is a topology optimisation loop in which the design variable is the level-set field, while the physical response is evaluated through periodical homogenisation.

## 2. Main components of the code

### 2.1 Periodic homogenisation

In [definition_3d.py](definition_3d.py), the code builds a periodic 3D unit-cell setting and solves representative cell problems under different macroscopic strain states. The resulting homogenised stiffness tensor is then used as the basis for the objective evaluation.

### 2.2 Level-set based design update

In [main_3d.py](main_3d.py), the optimisation loop uses the sensitivities obtained from the cost function, updates the level-set field, and performs a line-search / step-size control strategy to improve stability.

### 2.3 Configuration and numerical parameters

In [init_3d.py](init_3d.py), the user can tune:

- material properties such as Young's modulus and Poisson ratio;
- optimisation and line-search settings;
- volume-continuation / stage-control behaviour;
- mesh resolution and refinement strategy.

## 3. Current cost function setting

The current implementation is specifically set up for an uncoupled transverse isotropic (UTI) design objective.

The objective is constructed from representative stiffness components in a tetragonal / z-rot4 design space, including quantities such as:

- $C_{11}^{t}$, $C_{12}^{t}$, $C_{13}^{t}$, $C_{33}^{t}$, $C_{44}^{t}$, $C_{66}^{t}$;
- the UTI-related invariants $h_b$, $h_a$, $H$;
- a transverse-isotropy penalty term based on the residual $R_{TI}$.

The current objective is implemented in [definition_3d.py](definition_3d.py) through functions such as:

- `uti_ratio_term`
- `grad_uti_ratio_term`
- `ti_penalty_term`
- `grad_ti_penalty_term`
- `Phi_uti`
- `grad_Phi_uti`

In its current form, the total cost is approximately:

$$
J = J_{hb} + J_{ti}
$$

where

$$
J_{hb} = \beta_a h_b^2 + \beta_b / (h_a^2 + H^2 + \varepsilon)
$$

and $J_{ti}$ is a penalty term enforcing the transverse-isotropy relation.

## 4. Important note for other cases

This framework is currently configured for the UTI objective. If you want to optimise for a different material class or a different target property, the main optimisation machinery does not need to be rewritten. Instead, you should replace the cost function and its gradient in [definition_3d.py](definition_3d.py), especially the functions listed above.

In practice, for a new case you would:

1. define the new scalar objective from the homogenised stiffness tensor;
2. derive or implement the corresponding gradient with respect to the stiffness components;
3. replace the current UTI-related functions in [definition_3d.py](definition_3d.py);
4. keep the rest of the level-set update and homogenisation pipeline unchanged.

## 5. How to run

The main entry point is [main_3d.py](main_3d.py). After installing the required dependencies (especially FEniCS), run the script to start the optimisation loop.

## 6. Summary

This project provides a 3D topology optimisation framework for metamaterial design based on:

- periodic homogenisation;
- topological derivative / level-set update;
- a currently implemented UTI-specific cost function.

It is therefore well suited for UTI-type design targets, while remaining extensible to other material objectives by replacing the cost function module.
