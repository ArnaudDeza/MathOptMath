import gurobipy as gp
import numpy as np
from typing import Tuple, Optional

class GurobiSolver:
    """Wrapper class for Gurobi solver with common settings."""
    def __init__(self, time_limit: int = 300):
        self.time_limit = time_limit
        self.model = None
        
    def create_model(self, name: str) -> gp.Model:
        """Create a new Gurobi model with common settings."""
        self.model = gp.Model(name)
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('OutputFlag', 0)  # Suppress output
        return self.model

class TammesProblem:
    """Tammes Problem: Maximize minimum distance between points on a sphere."""
    def __init__(self, N: int):
        self.N = N
        self.solver = GurobiSolver()
        
    def solve(self) -> np.ndarray:
        """Solve the Tammes problem and return the points."""
        model = self.solver.create_model(f"Tammes_N{self.N}")
        
        # Variables: points on sphere (x, y, z coordinates)
        points = {}
        for i in range(self.N):
            points[i] = {
                'x': model.addVar(lb=-1, ub=1, name=f'x_{i}'),
                'y': model.addVar(lb=-1, ub=1, name=f'y_{i}'),
                'z': model.addVar(lb=-1, ub=1, name=f'z_{i}')
            }
        
        # Variable for minimum distance
        min_dist = model.addVar(lb=0, ub=2, name='min_dist')
        
        # Objective: maximize minimum distance
        model.setObjective(min_dist, gp.GRB.MAXIMIZE)
        
        # Constraints: points must lie on unit sphere
        for i in range(self.N):
            model.addConstr(
                points[i]['x']**2 + points[i]['y']**2 + points[i]['z']**2 == 1,
                name=f'sphere_{i}'
            )
        
        # Constraints: distance between any two points must be at least min_dist
        for i in range(self.N):
            for j in range(i+1, self.N):
                model.addConstr(
                    (points[i]['x'] - points[j]['x'])**2 +
                    (points[i]['y'] - points[j]['y'])**2 +
                    (points[i]['z'] - points[j]['z'])**2 >= min_dist**2,
                    name=f'dist_{i}_{j}'
                )
        
        # Solve
        model.optimize()
        
        if model.status == gp.GRB.OPTIMAL:
            # Extract solution
            solution = np.zeros((self.N, 3))
            for i in range(self.N):
                solution[i, 0] = points[i]['x'].X
                solution[i, 1] = points[i]['y'].X
                solution[i, 2] = points[i]['z'].X
            return solution
        else:
            raise RuntimeError(f"Failed to solve Tammes problem for N={self.N}")
            
    def get_objective(self, points: np.ndarray) -> float:
        """Calculate the minimum distance between points."""
        min_dist = float('inf')
        for i in range(self.N):
            for j in range(i+1, self.N):
                dist = np.linalg.norm(points[i] - points[j])
                min_dist = min(min_dist, dist)
        return min_dist

class CirclePackingProblem:
    """Equal Circle Packing Problem: Maximize radius of N equal circles in unit square."""
    def __init__(self, N: int):
        self.N = N
        self.solver = GurobiSolver()
        
    def solve(self) -> np.ndarray:
        """Solve the circle packing problem and return the centers."""
        model = self.solver.create_model(f"CirclePacking_N{self.N}")
        
        # Variables: circle centers (x, y coordinates)
        centers = {}
        for i in range(self.N):
            centers[i] = {
                'x': model.addVar(lb=0, ub=1, name=f'x_{i}'),
                'y': model.addVar(lb=0, ub=1, name=f'y_{i}')
            }
        
        # Variable for radius
        r = model.addVar(lb=0, ub=0.5, name='radius')
        
        # Objective: maximize radius
        model.setObjective(r, gp.GRB.MAXIMIZE)
        
        # Constraints: circles must be within unit square
        for i in range(self.N):
            model.addConstr(centers[i]['x'] >= r, name=f'left_{i}')
            model.addConstr(centers[i]['x'] <= 1-r, name=f'right_{i}')
            model.addConstr(centers[i]['y'] >= r, name=f'bottom_{i}')
            model.addConstr(centers[i]['y'] <= 1-r, name=f'top_{i}')
        
        # Constraints: circles must not overlap
        for i in range(self.N):
            for j in range(i+1, self.N):
                model.addConstr(
                    (centers[i]['x'] - centers[j]['x'])**2 +
                    (centers[i]['y'] - centers[j]['y'])**2 >= (2*r)**2,
                    name=f'overlap_{i}_{j}'
                )
        
        # Solve
        model.optimize()
        
        if model.status == gp.GRB.OPTIMAL:
            # Extract solution
            solution = np.zeros((self.N, 2))
            for i in range(self.N):
                solution[i, 0] = centers[i]['x'].X
                solution[i, 1] = centers[i]['y'].X
            return solution
        else:
            raise RuntimeError(f"Failed to solve Circle Packing problem for N={self.N}")
            
    def get_objective(self, centers: np.ndarray) -> float:
        """Calculate the maximum possible radius for the given centers."""
        # Calculate minimum distance between centers
        min_dist = float('inf')
        for i in range(self.N):
            for j in range(i+1, self.N):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, dist)
        
        # Calculate minimum distance to boundaries
        min_boundary = float('inf')
        for i in range(self.N):
            min_boundary = min(min_boundary,
                             centers[i, 0],  # distance to left
                             1 - centers[i, 0],  # distance to right
                             centers[i, 1],  # distance to bottom
                             1 - centers[i, 1])  # distance to top
        
        # Maximum radius is minimum of half the minimum center distance
        # and the minimum boundary distance
        return min(min_dist/2, min_boundary) 