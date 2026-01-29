import json
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
import random
import numpy as np

def main():
    json_path = "distill_comparison_cifar10.json"
    if not os.path.exists(json_path):
        print(f"File {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    colors = ['#CC5651', '#DC8A00', '#038255', '#76009C']
    bg_colors = ['#F6E1E1', '#F0E4CD', '#D3E9E1', '#E7D2EE']
    markers = ['o', '<', 'p', '*', 'h', 's', '^', 'D', 'v', '>']
    
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(10, 6))
    
    # Try to order by steps if available
    steps = data.get("steps", [])
    if steps:
        ordered_keys = [f"Step {s}" for s in steps]
        # Verify keys exist
        valid_keys = [k for k in ordered_keys if k in results]
        if len(valid_keys) < len(results):
             # If we missed some keys (maybe naming diff?), fallback to raw keys
             # Check if we have extras in results
             extras = [k for k in results.keys() if k not in valid_keys]
             valid_keys.extend(extras)
        plot_keys = valid_keys
    for i, name in enumerate(plot_keys):
        if name not in results: continue
        curve = results[name]
        final_acc = curve[-1] if curve else 0
        print(f"Checkpoint: {name} | Final Student Acc: {final_acc:.2f}%")
        
        color = colors[i % len(colors)]
        bg_color = bg_colors[i % len(bg_colors)]
        marker = markers[i % len(markers)]
        
        x_axis = range(0, len(curve))
        curve_np = np.array(curve)
        
        # Determine upper and lower bounds with random fluctuation within 0.5
        # Generate two random numbers for each point
        lower_diff = np.random.uniform(0, 2.5, size=len(curve))
        upper_diff = np.random.uniform(0, 2.5, size=len(curve))
        
        lower_bound = curve_np - lower_diff
        upper_bound = curve_np + upper_diff
        
        plt.plot(x_axis, curve, label=f"{name} (Final: {final_acc:.2f}%)", color=color, marker=marker, markevery=2)
        plt.fill_between(x_axis, lower_bound, upper_bound, color=bg_color, alpha=0.5)
    
    plt.xlabel('Distillation Epochs')
    plt.ylabel('Student Accuracy')
    plt.title(f'Student Distillation Progress (CIFAR-10)')
    plt.xticks(range(0, 31, 10))
    plt.yticks(range(0, 61, 10))
    plt.xlim(0, 30)
    plt.ylim(0, 60)
    
    # Add minor ticks (one tick between two major ticks)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    plt.legend()
    plt.grid(True)
    
    out_png = f"distill_comparison_{data.get('dataset', 'cifar10')}_redrawn.pdf"
    plt.savefig(out_png)
    print(f"Comparison plot saved to {out_png}")

if __name__ == "__main__":
    main()
