import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Optional, Tuple, List
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# Set style - use a built-in style instead of seaborn
plt.style.use('seaborn-v0_8')  # This is the correct style name for newer matplotlib versions

class PlotStyle:
    """Configuration class for plot styling."""
    def __init__(self,
                 title_fontsize: int = 16,
                 label_fontsize: int = 12,
                 tick_fontsize: int = 10,
                 title_font: str = 'DejaVu Sans',
                 label_font: str = 'DejaVu Sans',
                 dpi: int = 300,
                 save_dir: str = 'plots'):
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.title_font = title_font
        self.label_font = label_font
        self.dpi = dpi
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

def plot_tammes(points: np.ndarray, 
                style: Optional[PlotStyle] = None,
                save: bool = False,
                filename: str = 'tammes_solution.png') -> None:
    """Plot points on the unit sphere with enhanced visualization."""
    style = style or PlotStyle()
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    
    # Plot points with varying sizes based on z-coordinate
    sizes = 100 + 50 * (points[:, 2] + 1)  # Scale sizes based on z-coordinate
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=points[:, 2], cmap='viridis', s=sizes)
    
    # Plot sphere with transparency
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='gray', alpha=0.1)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax1, label='Z-coordinate')
    
    # Set labels and title
    ax1.set_xlabel('X', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_ylabel('Y', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_zlabel('Z', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_title('3D View', fontsize=style.title_fontsize, fontname=style.title_font)
    
    # Set equal aspect ratio
    ax1.set_box_aspect([1, 1, 1])
    
    # 2D projection plot
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                          cmap='viridis', s=sizes)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X', fontsize=style.label_fontsize, fontname=style.label_font)
    ax2.set_ylabel('Y', fontsize=style.label_fontsize, fontname=style.label_font)
    ax2.set_title('Top View', fontsize=style.title_fontsize, fontname=style.title_font)
    plt.colorbar(scatter2, ax=ax2, label='Z-coordinate')
    
    # Add main title
    fig.suptitle('Tammes Problem Solution', 
                 fontsize=style.title_fontsize + 4, 
                 fontname=style.title_font,
                 y=0.95)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(style.save_dir, filename), 
                   dpi=style.dpi, bbox_inches='tight')
    plt.show()

def plot_circle_packing(centers: np.ndarray, 
                       r: float,
                       style: Optional[PlotStyle] = None,
                       save: bool = False,
                       filename: str = 'circle_packing.png') -> None:
    """Plot circles in the unit square with enhanced visualization."""
    style = style or PlotStyle()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Basic circle packing
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2)
    
    # Create custom colormap for circles
    colors = plt.cm.plasma(np.linspace(0, 1, len(centers)))
    
    # Plot circles with varying colors
    for i, center in enumerate(centers):
        circle = plt.Circle(center, r, fill=False, color=colors[i], 
                          linewidth=2, alpha=0.7)
        ax1.add_patch(circle)
        # Add center point
        ax1.plot(center[0], center[1], 'o', color=colors[i], markersize=5)
    
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel('X', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_ylabel('Y', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_title('Circle Packing Solution', 
                  fontsize=style.title_fontsize, 
                  fontname=style.title_font)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance heatmap
    ax2.imshow(np.zeros((100, 100)), cmap='YlOrRd')
    for i, center1 in enumerate(centers):
        for j, center2 in enumerate(centers[i+1:], i+1):
            # Calculate distance between centers
            dist = np.linalg.norm(center1 - center2)
            # Plot line between centers
            ax2.plot([center1[0]*100, center2[0]*100], 
                    [center1[1]*100, center2[1]*100], 
                    'k-', alpha=0.3)
            # Add distance label
            mid = (center1 + center2) / 2
            ax2.text(mid[0]*100, mid[1]*100, f'{dist:.3f}', 
                    ha='center', va='center', fontsize=8)
    
    ax2.set_title('Center Distances', 
                  fontsize=style.title_fontsize, 
                  fontname=style.title_font)
    ax2.set_xlabel('X', fontsize=style.label_fontsize, fontname=style.label_font)
    ax2.set_ylabel('Y', fontsize=style.label_fontsize, fontname=style.label_font)
    
    # Add main title
    fig.suptitle(f'Circle Packing (r = {r:.4f})', 
                 fontsize=style.title_fontsize + 4, 
                 fontname=style.title_font,
                 y=0.95)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(style.save_dir, filename), 
                   dpi=style.dpi, bbox_inches='tight')
    plt.show()

def plot_performance(results: pd.DataFrame,
                    style: Optional[PlotStyle] = None,
                    save: bool = False,
                    filename: str = 'performance.png') -> None:
    """Plot runtime and objective scaling with enhanced visualization."""
    style = style or PlotStyle()
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Runtime plot
    ax1 = fig.add_subplot(221)
    sns.regplot(data=results, x='N', y='runtime', ax=ax1, 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    ax1.set_xlabel('N', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_ylabel('Runtime (s)', fontsize=style.label_fontsize, fontname=style.label_font)
    ax1.set_title('Runtime Scaling', 
                  fontsize=style.title_fontsize, 
                  fontname=style.title_font)
    ax1.grid(True, alpha=0.3)
    
    # Objective plot
    ax2 = fig.add_subplot(222)
    sns.regplot(data=results, x='N', y='objective', ax=ax2,
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    ax2.set_xlabel('N', fontsize=style.label_fontsize, fontname=style.label_font)
    ax2.set_ylabel('Objective Value', fontsize=style.label_fontsize, fontname=style.label_font)
    ax2.set_title('Objective Scaling', 
                  fontsize=style.title_fontsize, 
                  fontname=style.title_font)
    ax2.grid(True, alpha=0.3)
    
    # Runtime per N plot
    ax3 = fig.add_subplot(223)
    results['runtime_per_n'] = results['runtime'] / results['N']
    sns.barplot(data=results, x='N', y='runtime_per_n', ax=ax3)
    ax3.set_xlabel('N', fontsize=style.label_fontsize, fontname=style.label_font)
    ax3.set_ylabel('Runtime per N (s)', fontsize=style.label_fontsize, fontname=style.label_font)
    ax3.set_title('Runtime per N', 
                  fontsize=style.title_fontsize, 
                  fontname=style.title_font)
    ax3.grid(True, alpha=0.3)
    
    # Objective improvement plot
    ax4 = fig.add_subplot(224)
    results['objective_improvement'] = results['objective'].pct_change()
    sns.barplot(data=results, x='N', y='objective_improvement', ax=ax4)
    ax4.set_xlabel('N', fontsize=style.label_fontsize, fontname=style.label_font)
    ax4.set_ylabel('Objective Improvement (%)', 
                   fontsize=style.label_fontsize, 
                   fontname=style.label_font)
    ax4.set_title('Objective Improvement Rate', 
                  fontsize=style.title_fontsize, 
                  fontname=style.title_font)
    ax4.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle('Performance Analysis', 
                 fontsize=style.title_fontsize + 4, 
                 fontname=style.title_font,
                 y=0.95)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(style.save_dir, filename), 
                   dpi=style.dpi, bbox_inches='tight')
    plt.show()

def create_animation(results: pd.DataFrame,
                    style: Optional[PlotStyle] = None,
                    save: bool = False,
                    filename: str = 'animation.gif') -> None:
    """Create an animation showing the evolution of solutions."""
    from matplotlib.animation import FuncAnimation
    
    style = style or PlotStyle()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        N = results['N'].iloc[frame]
        objective = results['objective'].iloc[frame]
        runtime = results['runtime'].iloc[frame]
        
        # Create a circle packing visualization
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')
        r = objective
        centers = np.random.rand(N, 2) * (1 - 2*r) + r  # Placeholder centers
        
        for center in centers:
            circle = plt.Circle(center, r, fill=False, color='blue', alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f'N={N}, r={objective:.4f}, t={runtime:.2f}s',
                    fontsize=style.title_fontsize,
                    fontname=style.title_font)
    
    anim = FuncAnimation(fig, update, frames=len(results), 
                        interval=1000, repeat=True)
    
    if save:
        anim.save(os.path.join(style.save_dir, filename), 
                 writer='pillow', fps=1)
    plt.show() 