import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, MaxNLocator

def plot_bucket_metrics_evolution_combined(dfs, sizes, city, measure, scheme, bucketing_method, size, major_xticks=None, minor_xticks=None):
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bucket Metrics Evolution Analysis', fontsize=14, y=0.98)
    plt.figtext(0.5, 0.94, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=10, style='italic')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    metrics = [
        ('Avg Number of buckets', 'Avg Number of Buckets'),
        ('Avg Bucket size', 'Avg Bucket Size'),
        ('Avg Buckets with >1 Trajectory', 'Avg Buckets with >1 Trajectory'),
        ('Avg Largest Bucket Size', 'Avg Largest Bucket Size')
    ]
    if bucketing_method == "strict":
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}
        ]
    else:
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}
        ]

    for idx, (metric, title) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        for df, size in zip(dfs, sizes):
            df_filtered = df[df['Size'] == size]
            sns.lineplot(
                data=df_filtered, 
                x='Resolution', 
                y=metric, 
                hue='Layers',
                palette=colors,
                marker='o',
                markersize=5,
                linewidth=1.5,
                ax=axes[row, col]
            )
            if major_xticks is not None:
                axes[row, col].xaxis.set_major_locator(MultipleLocator(major_xticks))
            if minor_xticks is not None:
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(minor_xticks))
            axes[row, col].set_title(f'{title}', fontsize=10, pad=10)
            axes[row, col].set_xlabel('Grid Resolution (km)', fontsize=9)
            axes[row, col].set_ylabel(title, fontsize=9)
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            axes[row, col].tick_params(axis='both', which='major', labelsize=8)
            axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))
            if metric == 'Avg Number of buckets':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Bucket size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))               
            elif metric == 'Avg Buckets with >1 Trajectory':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Largest Bucket Size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            y_min, y_max = axes[row, col].get_ylim()
            padding = (y_max - y_min) * 0.1
            axes[row, col].set_ylim(y_min - padding, y_max + padding)
            legend = axes[row, col].legend(
                title='Layers',
                title_fontsize=8,
                fontsize=7,
                **legend_positions[idx]
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.08,
        left=0.08,
        right=0.98,
        hspace=0.4,
        wspace=0.2
    )
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    plt.show()
    
def plot_bucket_metrics_evolution_combined_disk(df, city, measure, scheme, bucketing_method, major_xticks=None, minor_xticks=None):
    """
    Create a comprehensive visualization with 4 rows (metrics) and 2 columns:
    - First column shows metrics vs diameter for different layers (aggregated over disks)
    - Second column shows metrics vs diameter for different disks (aggregated over layers)
    """
    fig, axes = plt.subplots(4, 2, figsize=(20, 30))
    fig.suptitle('Bucket Metrics Evolution Analysis: Layer-based vs Disk-based', fontsize=20, y=0.98)
    
    plt.figtext(0.5, 0.96, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()}',
                ha='center', fontsize=14, style='italic')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    metrics = [
        ('Avg Number of buckets', 'Avg Number of Buckets'),
        ('Avg Bucket size', 'Avg Bucket Size'),
        ('Avg Buckets with >1 Trajectory', 'Avg Buckets with >1 Trajectory'),
        ('Avg Largest Bucket Size', 'Avg Largest Bucket Size')
    ]
    
    if bucketing_method == "strict":
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},  # First row
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},   # Second row
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},   # Third row
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}    # Fourth row
        ]
    else:
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},  # First row
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},   # Second row
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},   # Third row
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}    # Fourth row
        ]
    
    for row, (metric, title) in enumerate(metrics):
        layer_data = df.groupby(['Diameter', 'Layers'])[metric].mean().reset_index()
        sns.lineplot(data=layer_data, 
                    x='Diameter', 
                    y=metric, 
                    hue='Layers',
                    palette=colors[:len(layer_data['Layers'].unique())],
                    marker='o',
                    markersize=7,
                    linewidth=2,
                    ax=axes[row, 0])
        
        disk_data = df.groupby(['Diameter', 'Disks'])[metric].mean().reset_index()
        sns.lineplot(data=disk_data, 
                    x='Diameter', 
                    y=metric, 
                    hue='Disks',
                    palette=colors[:len(disk_data['Disks'].unique())],
                    marker='o',
                    markersize=7,
                    linewidth=2,
                    ax=axes[row, 1])
        
        for col in range(2):
            if major_xticks is not None:
                axes[row, col].xaxis.set_major_locator(MultipleLocator(major_xticks))
            if minor_xticks is not None:
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(minor_xticks))
            
            axes[row, col].set_title(f'{title} vs {"Layers" if col == 0 else "Disks"}', fontsize=14, pad=20)
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=12)
            
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=12)
            else:
                axes[row, col].set_ylabel('')  # Remove y-label for other columns
            
            axes[row, col].tick_params(axis='both', which='major', labelsize=10)
            
            axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))
            
            if metric == 'Avg Number of buckets':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Bucket size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            elif metric == 'Avg Buckets with >1 Trajectory':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Largest Bucket Size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            y_min, y_max = axes[row, col].get_ylim()
            padding = (y_max - y_min) * 0.1
            axes[row, col].set_ylim(y_min - padding, y_max + padding)
            
            legend = axes[row, col].legend(
                title='Layers' if col == 0 else 'Disks',
                title_fontsize=8,
                fontsize=7,
                **legend_positions[row]
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.92,      # Space for the main title
        bottom=0.05,   # Space at the bottom
        left=0.1,      # Space on the left
        right=0.9,     # Space on the right
        hspace=0.3,    # Space between rows
        wspace=0.2     # Space between columns
    )
    
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    plt.show()

#Parameters and buckets analysis
def plot_bucket_metrics_vs_layers_multi_df(dfs, param_groups, major_ticks_list, city, measure, scheme, bucketing_method, minor_xticks=None):
    
    n_dfs = len(dfs)
    n_metrics = 4
    
    if len(param_groups) != n_dfs:
        raise ValueError("Length of param_groups must match number of dataframes")
    if len(major_ticks_list) != n_dfs:
        raise ValueError("Length of major_ticks_list must match number of dataframes")
    
    fig, axes = plt.subplots(n_dfs, n_metrics, figsize=(20, 6*n_dfs))
    fig.suptitle('Bucket Metrics Analysis Across Parameter Groups (vs layers)', fontsize=20, y=0.99)
    
    plt.figtext(0.5, 0.97, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: 1500',
                ha='center', fontsize=14, style='italic')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    metrics = [
        ('Avg Number of buckets', 'Avg Number of Buckets'),
        ('Avg Bucket size', 'Avg Bucket Size'),
        ('Avg Buckets with >1 Trajectory', 'Avg Buckets with >1 Trajectory'),
        ('Avg Largest Bucket Size', 'Avg Largest Bucket Size')
    ]
    
    if n_dfs == 1:
        axes = axes.reshape(1, -1)
    
    for row, (df, param_group, major_ticks) in enumerate(zip(dfs, param_groups, major_ticks_list)):
        for col, (metric, title) in enumerate(metrics):
            layer_data = df.groupby(['Diameter', 'Layers'])[metric].mean().reset_index()
            
            sns.lineplot(data=layer_data, 
                        x='Diameter', 
                        y=metric, 
                        hue='Layers',
                        palette=colors[:len(layer_data['Layers'].unique())],
                        marker='o',
                        markersize=7,
                        linewidth=2,
                        ax=axes[row, col])
            
            if major_ticks is not None:
                axes[row, col].xaxis.set_major_locator(MultipleLocator(major_ticks))
            if minor_xticks is not None:
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(minor_xticks))
            
            if row == 0:
                axes[row, col].set_title(title, fontsize=14, pad=20)
            
            if col == 0:
                axes[row, col].set_ylabel(f'{param_group}\n{title}', fontsize=12)
            else:
                axes[row, col].set_ylabel(title, fontsize=12)
            
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=12)
            
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
            axes[row, col].tick_params(axis='both', which='major', labelsize=10)
            
            axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))
            
            if metric == 'Avg Number of buckets':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Bucket size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            elif metric == 'Avg Buckets with >1 Trajectory':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Largest Bucket Size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            y_min, y_max = axes[row, col].get_ylim()
            padding = (y_max - y_min) * 0.1
            axes[row, col].set_ylim(y_min - padding, y_max + padding)
            
            
            if row == 1 and col > 0:
                legend_loc = 'upper left'
                
            
            else:  # All other plots
                legend_loc = 'upper left'
            
            legend = axes[row, col].legend(
                title='Layers',
                title_fontsize=8,
                fontsize=7,
                loc=legend_loc
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.93,      # Space for the main title
        bottom=0.1,    # Increased bottom space for x-axis labels
        left=0.1,      # Space on the left
        right=0.9,     # Space on the right
        hspace=0.3,    # Increased space between rows for x-axis labels
        wspace=0.3     # Space between columns
    )
    
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    plt.show()

def plot_bucket_metrics_vs_disks_multi_df(dfs, param_groups, major_ticks_list, city, measure, scheme, bucketing_method, minor_xticks=None):
   
    n_dfs = len(dfs)
    n_metrics = 4
    
    if len(param_groups) != n_dfs:
        raise ValueError("Length of param_groups must match number of dataframes")
    if len(major_ticks_list) != n_dfs:
        raise ValueError("Length of major_ticks_list must match number of dataframes")
    
    fig, axes = plt.subplots(n_dfs, n_metrics, figsize=(20, 6*n_dfs))
    fig.suptitle('Bucket Metrics Analysis Across Parameter Groups (vs disks)', fontsize=20, y=0.99)
    
    plt.figtext(0.5, 0.97, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: 1500',
                ha='center', fontsize=14, style='italic')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    metrics = [
        ('Avg Number of buckets', 'Avg Number of Buckets'),
        ('Avg Bucket size', 'Avg Bucket Size'),
        ('Avg Buckets with >1 Trajectory', 'Avg Buckets with >1 Trajectory'),
        ('Avg Largest Bucket Size', 'Avg Largest Bucket Size')
    ]
    
    if n_dfs == 1:
        axes = axes.reshape(1, -1)
    
    for row, (df, param_group, major_ticks) in enumerate(zip(dfs, param_groups, major_ticks_list)):
        for col, (metric, title) in enumerate(metrics):
            disk_data = df.groupby(['Diameter', 'Disks'])[metric].mean().reset_index()
            
            sns.lineplot(data=disk_data, 
                        x='Diameter', 
                        y=metric, 
                        hue='Disks',
                        palette=colors[:len(disk_data['Disks'].unique())],
                        marker='o',
                        markersize=7,
                        linewidth=2,
                        ax=axes[row, col])
            
            if major_ticks is not None:
                axes[row, col].xaxis.set_major_locator(MultipleLocator(major_ticks))
            if minor_xticks is not None:
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(minor_xticks))
            
            if row == 0:
                axes[row, col].set_title(title, fontsize=14, pad=20)
            
            if col == 0:
                axes[row, col].set_ylabel(f'{param_group}\n{title}', fontsize=12)
            else:
                axes[row, col].set_ylabel(title, fontsize=12)
            
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=12)
            
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
            axes[row, col].tick_params(axis='both', which='major', labelsize=10)
            
            axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))
            
            if metric == 'Avg Number of buckets':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Bucket size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            elif metric == 'Avg Buckets with >1 Trajectory':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Largest Bucket Size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            y_min, y_max = axes[row, col].get_ylim()
            padding = (y_max - y_min) * 0.1
            axes[row, col].set_ylim(y_min - padding, y_max + padding)
            
            
            if row == 0 or row >= 6:
                legend_loc = 'upper left'
                
            elif row == 2:
                legend_loc = 'upper left'
            
            elif row == 3:
                legend_loc = 'upper left'
            else:  # rows 1-5
                legend_loc = 'upper left'
            
            legend = axes[row, col].legend(
                title='Disks',
                title_fontsize=8,
                fontsize=7,
                loc=legend_loc
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')
            
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.93,      # Space for the main title
        bottom=0.05,   # Space at the bottom
        left=0.1,      # Space on the left
        right=0.9,     # Space on the right
        hspace=0.2,    # Space between rows
        wspace=0.3     # Space between columns
    )
    
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    plt.show()
    
### NEW functions for disks

def plot_bucket_metrics_DISK_evolution_combined(dfs, sizes, city, measure, scheme, bucketing_method, size, major_xticks=None, minor_xticks=None):
    """
    Create a comprehensive visualization with 2x2 grid of metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bucket Metrics Evolution Analysis', fontsize=14, y=0.98)
    plt.figtext(0.5, 0.94, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=10, style='italic')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    metrics = [
        ('Avg Number of buckets', 'Avg Number of Buckets'),
        ('Avg Bucket size', 'Avg Bucket Size'),
        ('Avg Buckets with >1 Trajectory', 'Avg Buckets with >1 Trajectory'),
        ('Avg Largest Bucket Size', 'Avg Largest Bucket Size')
    ]
    if bucketing_method == "strict":
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}
        ]
    else:
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}
        ]

    for idx, (metric, title) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        for df, size in zip(dfs, sizes):
            df_filtered = df[df['Size'] == size]
            
            # Aggregate across disks by taking the mean for each Diameter-Layers combination
            df_aggregated = df_filtered.groupby(['Diameter', 'Layers'])[metric].mean().reset_index()
            
            sns.lineplot(
                data=df_aggregated, 
                x='Diameter', 
                y=metric, 
                hue='Layers',
                palette=colors,
                marker='o',
                markersize=5,
                linewidth=1.5,
                ax=axes[row, col]
            )
            if major_xticks is not None:
                axes[row, col].xaxis.set_major_locator(MultipleLocator(major_xticks))
            if minor_xticks is not None:
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(minor_xticks))
            axes[row, col].set_title(f'{title}', fontsize=10, pad=10)
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=9)
            axes[row, col].set_ylabel(title, fontsize=9)
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            axes[row, col].tick_params(axis='both', which='major', labelsize=8)
            axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))
            if metric == 'Avg Number of buckets':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Bucket size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))     
            elif metric == 'Avg Buckets with >1 Trajectory':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Largest Bucket Size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            y_min, y_max = axes[row, col].get_ylim()
            padding = (y_max - y_min) * 0.1
            axes[row, col].set_ylim(y_min - padding, y_max + padding)
            legend = axes[row, col].legend(
                title='Layers',
                title_fontsize=8,
                fontsize=7,
                **legend_positions[idx]
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.08,
        left=0.08,
        right=0.98,
        hspace=0.4,
        wspace=0.2
    )
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    plt.show()
        
def plot_bucket_metrics_DISK_evolution_combined_DISKS(dfs, sizes, city, measure, scheme, bucketing_method, size, major_xticks=None, minor_xticks=None):
    """
    Create a comprehensive visualization with 2x2 grid of metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bucket Metrics Evolution Analysis', fontsize=14, y=0.98)
    plt.figtext(0.5, 0.94, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=10, style='italic')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    metrics = [
        ('Avg Number of buckets', 'Avg Number of Buckets'),
        ('Avg Bucket size', 'Avg Bucket Size'),
        ('Avg Buckets with >1 Trajectory', 'Avg Buckets with >1 Trajectory'),
        ('Avg Largest Bucket Size', 'Avg Largest Bucket Size')
    ]
    if bucketing_method == "strict":
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}
        ]
    else:
        legend_positions = [
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)},
            {'loc': 'upper right', 'bbox_to_anchor': (0.98, 0.98)},
            {'loc': 'upper left', 'bbox_to_anchor': (0.02, 0.98)}
        ]

    for idx, (metric, title) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        for df, size in zip(dfs, sizes):
            df_filtered = df[df['Size'] == size]
            
            # Aggregate across disks by taking the mean for each Diameter-Layers combination
            df_aggregated = df_filtered.groupby(['Diameter', 'Disks'])[metric].mean().reset_index()
            
            sns.lineplot(
                data=df_aggregated, 
                x='Diameter', 
                y=metric, 
                hue='Disks',
                palette=colors,
                marker='o',
                markersize=5,
                linewidth=1.5,
                ax=axes[row, col]
            )
            if major_xticks is not None:
                axes[row, col].xaxis.set_major_locator(MultipleLocator(major_xticks))
            if minor_xticks is not None:
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(minor_xticks))
            axes[row, col].set_title(f'{title}', fontsize=10, pad=10)
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=9)
            axes[row, col].set_ylabel(title, fontsize=9)
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            axes[row, col].tick_params(axis='both', which='major', labelsize=8)
            axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))
            if metric == 'Avg Number of buckets':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Bucket size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))     
            elif metric == 'Avg Buckets with >1 Trajectory':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            elif metric == 'Avg Largest Bucket Size':
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            y_min, y_max = axes[row, col].get_ylim()
            padding = (y_max - y_min) * 0.1
            axes[row, col].set_ylim(y_min - padding, y_max + padding)
            legend = axes[row, col].legend(
                title='Layers',
                title_fontsize=8,
                fontsize=7,
                **legend_positions[idx]
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.08,
        left=0.08,
        right=0.98,
        hspace=0.4,
        wspace=0.2
    )
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    plt.show()