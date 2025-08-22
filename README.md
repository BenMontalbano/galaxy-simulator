# galaxy-simulator 
*A GPU-accelerated N-body simulation of galactic dynamics and dark matter rotation curves*  

## Overview  
This project models the dynamics of a galaxy using an **N-body gravitational simulation**. It includes the effects of both baryonic matter and a **dark matter halo**, demonstrating how dark matter influences galactic rotation curves.  

The simulation leverages **GPU acceleration with CUDA (via Numba)** for high particle counts, and uses **Python scientific libraries** for numerical integration, visualization, and data analysis.  

This project is both a demonstration of astrophysical principles and an example of **applied simulation modeling**.  

---

## Features  
- **N-body gravitational dynamics** using leapfrog integration  
- **Dark matter halo modeling** for realistic galaxy rotation curves  
- **GPU acceleration** (CUDA with Numba) for performance  
- **Visualization** with VisPy for real-time galactic animations  
- **Rotation curve analysis** comparing galaxies with and without dark matter  

---

## Technology Stack  
- Python: NumPy, SciPy, Matplotlib, VisPy, PyQt5  
- CUDA via Numba for GPU acceleration  
- Scientific computing & numerical integration  


---

## Installation  
Clone the repository:  
```bash
git clone https://github.com/BenMontalbano/galaxy-simulator.git
cd galaxy-simulator
