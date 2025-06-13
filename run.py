import argparse
import pandas as pd
from problems import TammesProblem, CirclePackingProblem
from visualize import plot_tammes, plot_circle_packing, plot_performance
import time

def main():
    parser = argparse.ArgumentParser(description='Solve global optimization problems')
    parser.add_argument('--problem', choices=['tammes', 'circle'], required=True,
                      help='Problem to solve')
    parser.add_argument('--N', type=int, help='Number of points/circles')
    parser.add_argument('--plot', action='store_true', help='Plot the solution')
    parser.add_argument('--experiment', action='store_true', help='Run experiment with multiple N values')
    parser.add_argument('--Ns', nargs='+', type=int, help='List of N values for experiment')
    parser.add_argument('--out', help='Output file for experiment results')
    
    args = parser.parse_args()

    args.plot = True
    
    if args.experiment:
        if not args.Ns:
            parser.error("--experiment requires --Ns")
        if not args.out:
            parser.error("--experiment requires --out")
            
        results = []
        for N in args.Ns:
            print(f"\nSolving for N = {N}")
            if args.problem == 'tammes':
                problem = TammesProblem(N)
            else:
                problem = CirclePackingProblem(N)
                
            start_time = time.time()
            solution = problem.solve()
            runtime = time.time() - start_time
            
            results.append({
                'N': N,
                'runtime': runtime,
                'objective': problem.get_objective(solution)
            })
            
            if args.plot:
                if args.problem == 'tammes':
                    plot_tammes(solution)
                else:
                    plot_circle_packing(solution, problem.get_objective(solution))
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(args.out, index=False)
        print(f"\nResults saved to {args.out}")
        
        # Plot performance
        plot_performance(df)
        
    else:
        if not args.N:
            parser.error("Single solve requires --N")
            
        if args.problem == 'tammes':
            problem = TammesProblem(args.N)
        else:
            problem = CirclePackingProblem(args.N)
            
        solution = problem.solve()
        print(f"\nSolution found with objective value: {problem.get_objective(solution)}")
        
        if args.plot:
            if args.problem == 'tammes':
                plot_tammes(solution)
            else:
                plot_circle_packing(solution, problem.get_objective(solution))

if __name__ == '__main__':
    main() 


    '''
    python run.py --problem circle --N 3
    
    python run.py --problem tammes --N 10
    
    python run.py --problem circle --N 3 --experiment --Ns 3 4 5 6 7 8 9 10 --out results.csv

    python run.py --problem circle --experiment --Ns 3 4 5 6 7 --out results_circle.csv
    
    python run.py --problem tammes --experiment --Ns 3 4 5 6 7 8 9 10 --out results_tammes.csv
    
    '''