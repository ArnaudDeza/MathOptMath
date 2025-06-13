# Global Optimization Benchmarks

This project implements two classic global optimization problems using Gurobi:
1. **Tammes Problem**: Maximizing the minimum distance between N points on a unit sphere
2. **Equal-Circle Packing**: Maximizing the common radius of N equal circles packed in a unit square

## Requirements

The project requires Python 3.x and the following packages:
```
gurobipy
numpy
pandas
matplotlib
seaborn
```

You can install these dependencies using:
```bash
pip install -r requirements.txt
```

Note: You need a valid Gurobi license to run this code. Academic licenses are available for free.

## Project Structure

The codebase consists of four main files:

1. `problems.py`: Core optimization problem implementations
   - `GurobiSolver`: Wrapper class for Gurobi environment and settings
   - `TammesProblem`: Implementation of the Tammes problem
   - `CirclePackingProblem`: Implementation of the circle packing problem

2. `visualize.py`: Enhanced visualization functions
   - `PlotStyle`: Configuration class for plot styling
   - `plot_tammes`: 3D and 2D visualization of points on the unit sphere
   - `plot_circle_packing`: Circle packing visualization with distance analysis
   - `plot_performance`: Comprehensive performance analysis plots
   - `create_animation`: Solution evolution animation

3. `run.py`: Command-line interface for running experiments
   - Single solve mode: Solve one instance and optionally plot
   - Batch experiment mode: Run multiple instances and save results

4. `requirements.txt`: Project dependencies

## Usage

The project can be used in two modes:

### 1. Single Solve Mode

Solve a single instance of either problem:

```bash
# Solve Tammes problem with N=10 points
python run.py --problem tammes --N 10 --plot

# Solve circle packing with N=3 circles
python run.py --problem circle --N 3 --plot
```

Options:
- `--problem`: Choose between 'tammes' or 'circle'
- `--N`: Number of points/circles
- `--plot`: Optional flag to visualize the solution

### 2. Batch Experiment Mode

Run experiments over multiple N values:

```bash
# Run circle packing experiments
python run.py --problem circle --experiment --Ns 3 4 5 6 7 8 9 10 --out results.csv

# Run Tammes experiments
python run.py --problem tammes --experiment --Ns 10 20 30 40 50 --out results.csv
```

Options:
- `--experiment`: Enable batch experiment mode
- `--Ns`: List of N values to test
- `--out`: Output CSV file for results

## Visualization Features

The project includes enhanced visualization capabilities:

### Plot Styling
```python
from visualize import PlotStyle

# Customize plot appearance
style = PlotStyle(
    title_fontsize=16,
    label_fontsize=12,
    tick_fontsize=10,
    title_font='DejaVu Sans',
    label_font='DejaVu Sans',
    dpi=300,
    save_dir='plots'
)
```

### Tammes Problem Visualization
- 3D view with color-coded points based on z-coordinate
- 2D top view projection
- Interactive rotation in 3D view
- Automatic saving of plots

### Circle Packing Visualization
- Circle arrangement with color-coded circles
- Distance heatmap between circle centers
- Center point markers
- Distance labels between circles

### Performance Analysis
- Runtime scaling with regression line
- Objective value scaling
- Runtime per N analysis
- Objective improvement rate
- All plots with customizable styling

### Animation
- Solution evolution animation
- Frame-by-frame visualization
- Automatic saving as GIF

### Saving Plots
All visualization functions support automatic saving:
```python
# Save a plot
plot_tammes(points, style=style, save=True, filename='tammes_solution.png')

# Save performance analysis
plot_performance(results, style=style, save=True, filename='performance.png')

# Create and save animation
create_animation(results, style=style, save=True, filename='evolution.gif')
```

## Mathematical Formulation

### Tammes Problem
Maximize t subject to:
- ||x_i||² = 1 for all points i (points on unit sphere)
- ||x_i - x_j||² ≥ t² for all i < j (minimum distance constraint)
- Symmetry breaking: x₀ = (0,0,1) and z-coordinates ordered

### Circle Packing Problem
Maximize r subject to:
- (x_i - x_j)² + (y_i - y_j)² ≥ (2r)² for all i < j (non-overlap)
- r ≤ x_i ≤ 1-r and r ≤ y_i ≤ 1-r (containment)
- Symmetry breaking: x+y coordinates ordered

## Performance Considerations

- The problems are non-convex and solved as MIPs by Gurobi
- Symmetry breaking constraints are used to improve solver performance
- Runtime increases significantly with N
- For large N, consider adjusting the time limit in `GurobiSolver`

## Output

- Single solve mode prints N, objective value, and runtime
- Batch mode saves results to CSV with columns: N, objective, runtime
- Performance plots show runtime and objective scaling with N
- Solution visualizations show point/circle arrangements
- All plots can be saved in high resolution
- Animations show solution evolution

## License

This project requires a Gurobi license to run. Academic licenses are available for free at [Gurobi's website](https://www.gurobi.com/downloads/). 