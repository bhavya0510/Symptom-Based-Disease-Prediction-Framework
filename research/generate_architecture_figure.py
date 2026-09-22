"""
Generate Complete System Architecture Figure
Creates a visual representation of the actual pipeline implementation
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

RESEARCH_DIR = Path(__file__).resolve().parent
os.chdir(RESEARCH_DIR)

def create_architecture_figure():
    """Create complete system architecture figure"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define box positions and sizes
    box_width = 6
    box_height = 0.8
    x_center = 5
    y_start = 15
    y_spacing = 1.3
    
    # Define pipeline stages
    stages = [
        "Raw Symptom Dataset\n(4,962 records / 132 symptoms / 41 diseases)",
        "Cleaning & Deduplication\n(305 unique patterns)",
        "Leakage-Aware Split\n(Train/Val/Test: 70/15/15)",
        "Baseline Models\n(LR, SVM, RF, XGBoost, MLP)",
        "RF + XGBoost Ensemble\n(Soft-Voting)",
        "Probability Calibration\n(Isotonic/Sigmoid - selected on validation)",
        "Final Evaluation\n(Once on test set)",
        "Top-K Differential Diagnosis\n(1, 3, 5 candidates)",
        "Uncertainty Abstention\n(Confidence thresholding)",
        "XGBoost Component-Level SHAP\n(Local explanations)",
        "Sequential Question Selection\n(MI-based or Random baseline)",
        "Final Prediction + Explanation\n(Web API output)"
    ]
    
    # Define branching structure
    branches = {
        3: ["LR", "RF/SVM", "XGBoost/MLP"],  # Baseline Models split
        7: ["Top-K", "Abstention", "SHAP"],  # After calibration split
    }
    
    # Colors for different stages
    colors = {
        'data': '#E8F4F8',      # Light blue for data stages
        'model': '#FFF4E6',      # Light orange for model stages
        'evaluation': '#F0E8F8',  # Light purple for evaluation
        'output': '#E8F8E8'      # Light green for output
    }
    
    stage_colors = [
        colors['data'],    # Raw dataset
        colors['data'],    # Cleaning
        colors['data'],    # Split
        colors['model'],   # Baseline models
        colors['model'],   # Ensemble
        colors['model'],   # Calibration
        colors['evaluation'], # Final evaluation
        colors['output'],  # Top-K
        colors['output'],  # Abstention
        colors['output'],  # SHAP
        colors['output'],  # Sequential
        colors['output']   # Final output
    ]
    
    # Draw main pipeline boxes
    y_positions = []
    for i, (stage, color) in enumerate(zip(stages, stage_colors)):
        y = y_start - i * y_spacing
        y_positions.append(y)
        
        # Draw box
        box = FancyBboxPatch((x_center - box_width/2, y - box_height/2), 
                            box_width, box_height,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        
        # Add text
        ax.text(x_center, y, stage, ha='center', va='center', 
               fontsize=10, fontweight='bold', wrap=True)
    
    # Draw arrows between main stages
    for i in range(len(y_positions) - 1):
        # Skip arrows that branch
        if i == 2:  # After split, before baseline models
            continue
        if i == 6:  # After final evaluation, before outputs
            continue
            
        y1 = y_positions[i] - box_height/2
        y2 = y_positions[i+1] + box_height/2
        
        arrow = FancyArrowPatch((x_center, y1), (x_center, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Draw branching for baseline models
    baseline_y = y_positions[3]
    ensemble_y = y_positions[4]
    
    # Three branches from baseline models
    branch_x = [2.5, 5, 7.5]
    branch_labels = ["LR", "RF/SVM", "XGBoost/MLP"]
    
    for bx, label in zip(branch_x, branch_labels):
        # Draw branch box
        box = FancyBboxPatch((bx - 1, baseline_y - 0.4), 
                            2, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['model'],
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)
        ax.text(bx, baseline_y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        # Arrow from main baseline box to branch
        arrow = FancyArrowPatch((x_center, baseline_y + box_height/2), 
                               (bx, baseline_y + 0.4),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray')
        ax.add_patch(arrow)
        
        # Arrow from branch to ensemble
        arrow = FancyArrowPatch((bx, baseline_y - 0.4), 
                               (x_center, ensemble_y + box_height/2),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    # Draw branching for outputs
    output_y = y_positions[7]
    eval_y = y_positions[6]
    
    output_x = [2, 5, 8]
    output_labels = ["Top-K", "Abstention", "SHAP"]
    
    for ox, label in zip(output_x, output_labels):
        # Draw branch box
        box = FancyBboxPatch((ox - 1, output_y - 0.4), 
                            2, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['output'],
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)
        ax.text(ox, output_y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        # Arrow from evaluation to branch
        arrow = FancyArrowPatch((x_center, eval_y - box_height/2), 
                               (ox, output_y + 0.4),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    # Add arrows from outputs to sequential questioning
    sequential_y = y_positions[10]
    for ox in output_x:
        arrow = FancyArrowPatch((ox, output_y - 0.4), 
                               (x_center, sequential_y + box_height/2),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    # Add title
    ax.text(x_center, 15.8, "Complete System Architecture", 
           ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor='black', label='Data Processing'),
        mpatches.Patch(facecolor=colors['model'], edgecolor='black', label='Model Training'),
        mpatches.Patch(facecolor=colors['evaluation'], edgecolor='black', label='Evaluation'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output Components')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/system_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/system_architecture.pdf', bbox_inches='tight')
    print("✅ Saved system architecture figure to results/system_architecture.png and results/system_architecture.pdf")
    
    return fig

if __name__ == "__main__":
    print("Generating complete system architecture figure...")
    create_architecture_figure()
    print("✅ Architecture figure generation complete")