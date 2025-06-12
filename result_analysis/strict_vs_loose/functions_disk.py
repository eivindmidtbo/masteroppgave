import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, MaxNLocator

def plot_mean_metrics_vs_diameter_new(df, city, scheme, measure, bucketing_method, threshold, size, tick):
    """
    Plots the mean Precision, Recall, and F1 Score vs. Diameter,
    averaging over all Layers and Disks, with consistent styling.
    """
    mean_df = df.groupby('Diameter')[['Avg Precision', 'Avg Recall', 'Avg F1 Score']].mean().reset_index()
    mean_df_melted = mean_df.melt(id_vars='Diameter',
                                  value_vars=['Avg Precision', 'Avg Recall', 'Avg F1 Score'],
                                  var_name='Metric', value_name='Score')
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=mean_df_melted, x='Diameter', y='Score', hue='Metric',
                 palette=palette, marker='o', ax=ax, legend=True)
    
    fig.suptitle('Mean Metrics Across Diameters', fontsize=20, y=1)
    plt.figtext(0.5, 0.85, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | '
                            f'Bucketing Method: {bucketing_method.upper()} | Threshold: {threshold} | '
                            f'Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    legend = ax.get_legend()
    legend.set_title('Metric')
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_boxstyle('round,pad=0.2')
    legend.set_bbox_to_anchor((0.89, 0.5))
    legend._loc = 10  # center inside axes

    ax.set_xlabel('Disk Diameter (km)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(tick))
    ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.80,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.3,
        wspace=0.2
    )
    
    plt.show()
    
def plot_mean_metrics_vs_diameter_comparison(dfs, labels, city, scheme, measure, bucketing_method, threshold, size, ticks):
    
    if len(dfs) != len(labels) or len(dfs) != len(ticks):
        raise ValueError("The number of dataframes, labels, and ticks must match")
    
    n_plots = len(dfs)
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]  # Make axes iterable if there's only one plot
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (ax, df, label, tick) in enumerate(zip(axes, dfs, labels, ticks)):
        mean_df = df.groupby('Diameter')[['Avg Precision', 'Avg Recall', 'Avg F1 Score']].mean().reset_index()
        mean_df_melted = mean_df.melt(id_vars='Diameter',
                                     value_vars=['Avg Precision', 'Avg Recall', 'Avg F1 Score'],
                                     var_name='Metric', value_name='Score')
        
        sns.lineplot(data=mean_df_melted, x='Diameter', y='Score', hue='Metric',
                    palette=palette, marker='o', ax=ax, legend=True)
        
        ax.set_title(f'{label}', fontsize=16, pad=20)
        
        legend = ax.get_legend()
        legend.set_title('Metric')
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_boxstyle('round,pad=0.2')
        
        if idx == 0:
            legend_location = 'upper left'
            legend.set_loc(legend_location)
        else:
            legend.set_bbox_to_anchor((0.89, 0.5))
            legend._loc = 10  # center inside axes
        
        y_min = mean_df_melted['Score'].min()
        y_max = mean_df_melted['Score'].max()
        y_padding = (y_max - y_min) * 0.1  # 10% padding
        y_min = max(0, y_min - y_padding)
        
        if idx == 0:
            y_max = min(1.0, y_max + y_padding * 3)  # Triple the padding for first figure
        else:
            y_max = min(1.0, y_max + y_padding)
        
        y_range = y_max - y_min
        if y_range <= 0.2:
            major_tick = 0.02
            minor_tick = 0.01
        elif y_range <= 0.5:
            major_tick = 0.05
            minor_tick = 0.025
        else:
            major_tick = 0.1
            minor_tick = 0.05
            
        ax.set_xlabel('Disk Diameter (km)', fontsize=12)
        if idx == 0:
            ax.set_ylabel(f'Threshold: {threshold}\nScore', fontsize=12)
        else:
            ax.set_ylabel('Score', fontsize=12)
            
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(major_tick))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_tick))
            
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_facecolor('white')
    
    fig.suptitle('Mean Metrics Across Disk Diameters', fontsize=20, y=1.05)
    plt.figtext(0.5, 0.97, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | '
                          f'Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    fig.patch.set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.3,
        wspace=0.3
    )
    
    plt.show()
    
    
def plot_mean_metrics_vs_diameter_multi_threshold(dfs, labels, thresholds, city, scheme, measure, bucketing_method, size, ticks):
    
    if len(dfs) != len(labels) or len(dfs) != len(ticks):
        raise ValueError("The number of dataframes, labels, and ticks must match")
    
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for df in dfs:
        for threshold in thresholds:
            df_threshold = df[df['Threshold'] == threshold]
            if not df_threshold.empty:
                mean_df = df_threshold.groupby('Diameter')[['Avg Precision', 'Avg Recall', 'Avg F1 Score']].mean().reset_index()
                mean_df_melted = mean_df.melt(id_vars='Diameter',
                                             value_vars=['Avg Precision', 'Avg Recall', 'Avg F1 Score'],
                                             var_name='Metric', value_name='Score')
                global_y_min = min(global_y_min, mean_df_melted['Score'].min())
                global_y_max = max(global_y_max, mean_df_melted['Score'].max())
    
    y_padding = (global_y_max - global_y_min) * 0.1  # 10% padding
    global_y_min = max(0, global_y_min - y_padding)
    global_y_max = min(1.0, global_y_max + y_padding)
    
    y_range = global_y_max - global_y_min
    if y_range <= 0.2:
        major_tick = 0.02
        minor_tick = 0.01
    elif y_range <= 0.5:
        major_tick = 0.05
        minor_tick = 0.025
    else:
        major_tick = 0.1
        minor_tick = 0.05

    n_thresholds = len(thresholds)
    n_plots_per_row = len(dfs)
    
    fig = plt.figure(figsize=(16, 6 * n_thresholds))
    gs = fig.add_gridspec(n_thresholds, n_plots_per_row, width_ratios=[1, 1])
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for row, threshold in enumerate(thresholds):
        for col, (df, label, tick) in enumerate(zip(dfs, labels, ticks)):
            ax = fig.add_subplot(gs[row, col])
            
            df_threshold = df[df['Threshold'] == threshold]
            
            mean_df = df_threshold.groupby('Diameter')[['Avg Precision', 'Avg Recall', 'Avg F1 Score']].mean().reset_index()
            mean_df_melted = mean_df.melt(id_vars='Diameter',
                                         value_vars=['Avg Precision', 'Avg Recall', 'Avg F1 Score'],
                                         var_name='Metric', value_name='Score')
            
            sns.lineplot(data=mean_df_melted, x='Diameter', y='Score', hue='Metric',
                        palette=palette, marker='o', ax=ax, legend=(col == 0))
            
            if row == 0:
                ax.set_title(f'{label}', fontsize=16, pad=20)
            
            if col == 0:
                legend = ax.get_legend()
                legend.set_title('Metric')
                legend.get_frame().set_alpha(0.8)
                legend.get_frame().set_boxstyle('round,pad=0.2')
            
            ax.set_xlabel('Disk Diameter (km)', fontsize=12)
            
            if col == 0:
                ax.set_ylabel(f'Threshold: {threshold}\nScore', fontsize=12)
            else:
                ax.set_ylabel('Score', fontsize=12)
            
            ax.set_ylim(global_y_min, global_y_max)
            ax.yaxis.set_major_locator(MultipleLocator(major_tick))
            ax.yaxis.set_minor_locator(MultipleLocator(minor_tick))
            
            ax.xaxis.set_major_locator(MultipleLocator(tick))
            ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.set_facecolor('white')
    
    fig.suptitle('Mean Metrics Across Diameters for Different Thresholds', fontsize=20, y=1.005)
    plt.figtext(0.5, 0.99, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | '
                          f'Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    fig.patch.set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.96,
        bottom=0.05,
        left=0.1,
        right=0.9,
        hspace=0.4,
        wspace=0.3
    )
    
    plt.show()
    

    
    
    
#Paramater effect on metrics
def plot_metrics_vs_diameter_layers_multi_df_2(dfs, param_groups, major_ticks_list, city, measure, scheme, bucketing_method, size):

    n_dfs = len(dfs)
    n_metrics = 3  # Precision, Recall, F1-score
    
    if len(param_groups) != n_dfs:
        raise ValueError("Length of param_groups must match number of dataframes")
    if len(major_ticks_list) != n_dfs:
        raise ValueError("Length of major_ticks_list must match number of dataframes")
    
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    ax_top_left = plt.subplot(gs[0, 0])  # Top left (Precision)
    ax_top_right = plt.subplot(gs[0, 1])  # Top right (Recall)
    ax_bottom_left = plt.subplot(gs[1, 0])  # Bottom left (F1 Score)
    ax_bottom_right = plt.subplot(gs[1, 1])  # Bottom right (empty)
    ax_bottom_right.axis('off')  # Hide the empty axis
    
    axes = [ax_top_left, ax_top_right, ax_bottom_left]
    metric_info = [
        ('Avg Precision', 'Precision', 'Precision Score'),
        ('Avg Recall', 'Recall', 'Recall Score'),
        ('Avg F1 Score', 'F1 Score', 'F1 Score')
    ]
    
    fig.suptitle('Layer Count Effect on Metrics Across Disk Diameters', fontsize=20, y=0.99)
    plt.figtext(0.5, 0.94, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    df, param_group, major_ticks = dfs[0], param_groups[0], major_ticks_list[0]
    layer_data_all = df.groupby(['Diameter', 'Layers'])['Avg Precision'].mean().reset_index()
    dia_min = layer_data_all['Diameter'].min()
    dia_max = layer_data_all['Diameter'].max()
    dia_range_str = f"{dia_min:.1f}–{dia_max:.1f}"
    for col, (metric, title, ylabel) in enumerate(metric_info):
        layer_data = df.groupby(['Diameter', 'Layers'])[metric].mean().reset_index()
        
        sns.lineplot(data=layer_data, 
                    x='Diameter', 
                    y=metric,
                    hue='Layers',
                    palette=colors[:len(layer_data['Layers'].unique())],
                    marker='o',
                    markersize=7,
                    linewidth=2,
                    ax=axes[col])
        
        if major_ticks is not None:
            axes[col].xaxis.set_major_locator(MultipleLocator(major_ticks))
            axes[col].xaxis.set_minor_locator(MultipleLocator(major_ticks/2))
        
        if col == 0:  # Precision
            y_min, y_max = layer_data[metric].min(), layer_data[metric].max()
            padding = (y_max - y_min) * 0.1
            y_min = max(0, y_min - padding)
            y_max = min(1, y_max + padding)
            axes[col].yaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            axes[col].set_ylim(0, y_max)
            axes[col].yaxis.set_tick_params(labelleft=True)
            axes[col].tick_params(axis='y', which='major', left=True, labelleft=True)
            axes[col].set_ylabel(f"{dia_range_str}\n{ylabel}", fontsize=12)
        elif col == 2:  # F1 Score (bottom left)
            y_min, y_max = layer_data[metric].min(), layer_data[metric].max()
            padding = (y_max - y_min) * 0.1
            y_min = max(0, y_min - padding)
            y_max = min(1, y_max + padding)
            axes[col].yaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            axes[col].set_ylim(0, y_max)
            axes[col].yaxis.set_tick_params(labelleft=True)
            axes[col].tick_params(axis='y', which='major', left=True, labelleft=True)
            axes[col].set_ylabel(f"{dia_range_str}\n{ylabel}", fontsize=12)
        else:  # Recall (top right)
            axes[col].yaxis.set_major_locator(MultipleLocator(0.1))
            axes[col].yaxis.set_minor_locator(MultipleLocator(0.05))
            axes[col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            y_min, y_max = axes[col].get_ylim()
            y_min_rounded = np.floor(y_min * 10) / 10
            y_max_rounded = np.ceil(y_max * 10) / 10
            axes[col].set_ylim(y_min_rounded, y_max_rounded)    
            axes[col].set_ylabel(ylabel, fontsize=12)
        
        axes[col].set_title(title, fontsize=14, pad=20)
        axes[col].set_xlabel('Disk Diameter (km)', fontsize=12)
        axes[col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        if col == 0:
            legend_loc = 'upper right'
        elif col == 1:
            legend_loc = 'lower right'
        else:
            legend_loc = 'upper right'
        legend = axes[col].legend(
            title='Layers',
            title_fontsize=10,
            fontsize=9,
            loc=legend_loc
        )
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_boxstyle('round,pad=0.2')
        plt.setp(axes[col].get_yticklabels(), visible=True)
        axes[col].yaxis.set_tick_params(which='both', length=4, width=1, direction='out')
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.08,
        left=0.15,
        right=0.9,
        hspace=0.4,
        wspace=0.2
    )
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes:
        ax.set_facecolor('white')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
def plot_metrics_vs_diameter_disks_multi_df(dfs, param_groups, major_ticks_list, city, measure, scheme, bucketing_method, size):
    
    n_dfs = len(dfs)
    n_metrics = 3  # Precision, Recall, F1-score
    
    if len(param_groups) != n_dfs:
        raise ValueError("Length of param_groups must match number of dataframes")
    if len(major_ticks_list) != n_dfs:
        raise ValueError("Length of major_ticks_list must match number of dataframes")
    
    fig, axes = plt.subplots(n_dfs, n_metrics, figsize=(35, 6*n_dfs))
    fig.suptitle('Effect of Number of Disks on Performance Metrics', fontsize=20, y=0.99)
    
    plt.figtext(0.5, 0.95, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    metrics = [
        ('Avg Precision', 'Precision'),
        ('Avg Recall', 'Recall'),
        ('Avg F1 Score', 'F1 Score')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d',
              '#9edae5', '#c7c7c7', '#98df8a', '#ffbb78', '#aec7e8']
    
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
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(major_ticks/2))
            
            if row == 0:  # First row
                y_min, y_max = disk_data[metric].min(), disk_data[metric].max()
                padding = (y_max - y_min) * 0.1
                y_min = max(0, y_min - padding)
                y_max = min(1, y_max + padding)
                
                axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins=6))
                axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                
                axes[row, col].set_ylim(0, y_max)
                
                axes[row, col].yaxis.set_tick_params(labelleft=True)
                axes[row, col].tick_params(axis='y', which='major', left=True, labelleft=True)
                
            else:  # Second row
                axes[row, col].yaxis.set_major_locator(MultipleLocator(0.1))
                axes[row, col].yaxis.set_minor_locator(MultipleLocator(0.05))
                axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                
                y_min, y_max = axes[row, col].get_ylim()
                y_min_rounded = np.floor(y_min * 10) / 10
                y_max_rounded = np.ceil(y_max * 10) / 10
                axes[row, col].set_ylim(y_min_rounded, y_max_rounded)
            
            if row == 0:
                axes[row, col].set_title(title, fontsize=14, pad=20)
            
            if col == 0:
                axes[row, col].set_ylabel(f'{param_group}\n{title} Score', fontsize=12)
                
            elif col == 2:
                axes[row, col].set_ylabel(f"{title}", fontsize=12)
            else:
                axes[row, col].set_ylabel(f"{title} Score", fontsize=12)
            
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=12)
            
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
            if row == 0 and col == 0:
                legend_loc = 'upper left'
            elif row == 0 and col == 1: 
                legend_loc = 'upper left'
            elif row == 0 and col == 2:
                    legend_loc = 'upper left'
            elif row == 1 and col == 0:  # Second row, middle column - bottom left
                legend_loc = 'upper left'
            elif row == 1 and col == 1:  # Second row, middle column - bottom left
                legend_loc = 'lower left'
            elif row == 1 and col == 2:  # Second row, middle column - bottom left
                legend_loc = 'upper left'
            else:  # All other plots
                legend_loc = 'center right'

            legend = axes[row, col].legend(
                title='Disks',
                title_fontsize=10,
                fontsize=9,
                loc=legend_loc
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')
            
            plt.setp(axes[row, col].get_yticklabels(), visible=True)
            axes[row, col].yaxis.set_tick_params(which='both', length=4, width=1, direction='out')
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.87,      # Space for the main title
        bottom=0.1,    # Slightly increased bottom space for x-axis labels
        left=0.12,     # Increased left margin for y-axis labels
        right=0.95,    # Space on the right
        hspace=0.3,    # Increased space between rows for x-axis labels
        wspace=0.2     # Space between columns
    )
    
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    plt.show()
    



#Paramater effect on metrics
def plot_metrics_vs_diameter_layers_multi_df(dfs, param_groups, major_ticks_list, city, measure, scheme, bucketing_method, size):
    
    n_dfs = len(dfs)
    n_metrics = 3  # Precision, Recall, F1-score
    
    if len(param_groups) != n_dfs:
        raise ValueError("Length of param_groups must match number of dataframes")
    if len(major_ticks_list) != n_dfs:
        raise ValueError("Length of major_ticks_list must match number of dataframes")
    
    fig, axes = plt.subplots(n_dfs, n_metrics, figsize=(35, 6*n_dfs))
    fig.suptitle('Effect of Number of Layers on Performance Metrics', fontsize=20, y=0.99)
    
    plt.figtext(0.5, 0.95, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    metrics = [
        ('Avg Precision', 'Precision'),
        ('Avg Recall', 'Recall'),
        ('Avg F1 Score', 'F1 Score')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
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
                axes[row, col].xaxis.set_minor_locator(MultipleLocator(major_ticks/2))
            
            if row == 0:  # First row
                y_min, y_max = layer_data[metric].min(), layer_data[metric].max()
                padding = (y_max - y_min) * 0.1
                y_min = max(0, y_min - padding)
                y_max = min(1, y_max + padding)
                
                axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins=6))
                axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                
                axes[row, col].set_ylim(0, y_max)
                axes[row, col].yaxis.set_tick_params(labelleft=True)
                axes[row, col].tick_params(axis='y', which='major', left=True, labelleft=True)
                
            else:  # Second row
                axes[row, col].yaxis.set_major_locator(MultipleLocator(0.1))
                axes[row, col].yaxis.set_minor_locator(MultipleLocator(0.05))
                axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                
                y_min, y_max = axes[row, col].get_ylim()
                y_min_rounded = np.floor(y_min * 10) / 10
                y_max_rounded = np.ceil(y_max * 10) / 10
                axes[row, col].set_ylim(y_min_rounded, y_max_rounded)
            
            if row == 0:
                axes[row, col].set_title(title, fontsize=14, pad=20)
            
            if col == 0:
                axes[row, col].set_ylabel(f'{param_group}\n{title}', fontsize=12)
            else:
                axes[row, col].set_ylabel(title, fontsize=12)
            
            axes[row, col].set_xlabel('Disk Diameter (km)', fontsize=12)
            
            axes[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
           
            if row == 0 and col == 0:
                legend_loc = 'lower left'
            elif row == 0 and col == 1: 
                legend_loc = 'upper left'
            elif row == 0 and col == 2:
                    legend_loc = 'lower left'
            elif row == 1 and col == 1:  # Second row, middle column - bottom left
                legend_loc = 'lower right'
            elif row == 1 and col == 2:  # Second row, middle column - bottom left
                legend_loc = 'upper right'
            else:  # All other plots
                legend_loc = 'upper left'
            
            legend = axes[row, col].legend(
                title='Layers',
                title_fontsize=10,
                fontsize=9,
                loc=legend_loc
            )
            legend.get_frame().set_alpha(0.8)
            legend.get_frame().set_boxstyle('round,pad=0.2')
            
            plt.setp(axes[row, col].get_yticklabels(), visible=True)
            axes[row, col].yaxis.set_tick_params(which='both', length=4, width=1, direction='out')
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.87,      # Space for the main title
        bottom=0.1,    # Slightly increased bottom space for x-axis labels
        left=0.12,     # Increased left margin for y-axis labels
        right=0.95,    # Space on the right
        hspace=0.3,    # Increased space between rows for x-axis labels
        wspace=0.2     # Space between columns
    )
    
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    plt.show()
    

def plot_metrics_vs_diameter_disks_multi_df_2(dfs, param_groups, major_ticks_list, city, measure, scheme, bucketing_method, size):
    
    n_dfs = len(dfs)
    n_metrics = 3  # Precision, Recall, F1-score
    
    if len(param_groups) != n_dfs:
        raise ValueError("Length of param_groups must match number of dataframes")
    if len(major_ticks_list) != n_dfs:
        raise ValueError("Length of major_ticks_list must match number of dataframes")
    
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    ax_top_left = plt.subplot(gs[0, 0])  # Top left (Precision)
    ax_top_right = plt.subplot(gs[0, 1])  # Top right (Recall)
    ax_bottom_left = plt.subplot(gs[1, 0])  # Bottom left (F1 Score)
    ax_bottom_right = plt.subplot(gs[1, 1])  # Bottom right (empty)
    ax_bottom_right.axis('off')  # Hide the empty axis
    
    axes = [ax_top_left, ax_top_right, ax_bottom_left]
    metric_info = [
        ('Avg Precision', 'Precision', 'Precision Score'),
        ('Avg Recall', 'Recall', 'Recall Score'),
        ('Avg F1 Score', 'F1 Score', 'F1 Score')
    ]
    
    fig.suptitle('Disk Count Effect on Metrics Across Disk Diameters', fontsize=20, y=0.99)
    plt.figtext(0.5, 0.94, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d',
              '#9edae5', '#c7c7c7', '#98df8a', '#ffbb78', '#aec7e8']
    
    df, param_group, major_ticks = dfs[0], param_groups[0], major_ticks_list[0]
    disk_data_all = df.groupby(['Diameter', 'Disks'])['Avg Precision'].mean().reset_index()
    dia_min = disk_data_all['Diameter'].min()
    dia_max = disk_data_all['Diameter'].max()
    dia_range_str = f"{dia_min:.2f}–{dia_max:.2f}"
    for col, (metric, title, ylabel) in enumerate(metric_info):
        disk_data = df.groupby(['Diameter', 'Disks'])[metric].mean().reset_index()
        
        sns.lineplot(data=disk_data, 
                    x='Diameter', 
                    y=metric,
                    hue='Disks',
                    palette=colors[:len(disk_data['Disks'].unique())],
                    marker='o',
                    markersize=7,
                    linewidth=2,
                    ax=axes[col])
        
        if major_ticks is not None:
            axes[col].xaxis.set_major_locator(MultipleLocator(major_ticks))
            axes[col].xaxis.set_minor_locator(MultipleLocator(major_ticks/2))
        
        if col == 0:  # Precision
            y_min, y_max = disk_data[metric].min(), disk_data[metric].max()
            padding = (y_max - y_min) * 0.1
            y_min = max(0, y_min - padding)
            y_max = min(1, y_max + padding)
            axes[col].yaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            axes[col].set_ylim(0, y_max)
            axes[col].yaxis.set_tick_params(labelleft=True)
            axes[col].tick_params(axis='y', which='major', left=True, labelleft=True)
            axes[col].set_ylabel(f"{dia_range_str}\n{ylabel}", fontsize=12)
        elif col == 2:  # F1 Score (bottom left)
            y_min, y_max = disk_data[metric].min(), disk_data[metric].max()
            padding = (y_max - y_min) * 0.1
            y_min = max(0, y_min - padding)
            y_max = min(1, y_max + padding)
            axes[col].yaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            axes[col].set_ylim(0, y_max)
            axes[col].yaxis.set_tick_params(labelleft=True)
            axes[col].tick_params(axis='y', which='major', left=True, labelleft=True)
            axes[col].set_ylabel(f"{dia_range_str}\n{ylabel}", fontsize=12)
        else:  # Recall (top right)
            axes[col].yaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[col].yaxis.set_major_locator(MultipleLocator(0.2))
            axes[col].yaxis.set_minor_locator(MultipleLocator(0.1))
            axes[col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            y_min, y_max = axes[col].get_ylim()
            y_min_rounded = np.floor(y_min * 10) / 10
            y_max_rounded = np.ceil(y_max * 10) / 10
            axes[col].set_ylim(y_min_rounded, y_max_rounded)
            axes[col].set_ylabel(ylabel, fontsize=12)
        
        axes[col].set_title(title, fontsize=14, pad=20)
        axes[col].set_xlabel('Disk Diameter (km)', fontsize=12)
        axes[col].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Legend positions
        if col == 0:
            legend_loc = 'lower right'
        elif col == 1:
            legend_loc = 'upper right'
        else:
            legend_loc = 'lower right'
        legend = axes[col].legend(
            title='Disks',
            title_fontsize=10,
            fontsize=9,
            loc=legend_loc
        )
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_boxstyle('round,pad=0.2')
        plt.setp(axes[col].get_yticklabels(), visible=True)
        axes[col].yaxis.set_tick_params(which='both', length=4, width=1, direction='out')
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.85,
        bottom=0.08,
        left=0.15,
        right=0.9,
        hspace=0.4,
        wspace=0.2
    )
    fig.patch.set_facecolor('#f0f0f0')
    for ax in axes:
        ax.set_facecolor('white')
    plt.show()

