import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, MaxNLocator

def plot_mean_values_grid_sizes(dfs, city, scheme, measure, bucketing_method, thresholds, sizes, tick):
    fig, axes = plt.subplots(len(thresholds), len(sizes), figsize=(10 * len(sizes), 5 * len(thresholds)))

    if len(thresholds) == 1:
        axes = [axes]
    if len(sizes) == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle('Mean Metrics Across Grid Resolutions for Different Data Sizes', fontsize=20, y=0.99)
    plt.figtext(0.5, 0.90, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | Bucketing Method: {bucketing_method.upper()}',
                ha='center', fontsize=14, style='italic')

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for row_idx, threshold in enumerate(thresholds):
        for col_idx, (df, size) in enumerate(zip(dfs, sizes)):
            ax = axes[row_idx][col_idx]

            df_threshold = df[df['Threshold'] == threshold]
            if df_threshold.empty:
                ax.set_visible(False)
                continue

            mean_df = df_threshold.groupby('Resolution')[['Avg Precision', 'Avg Recall', 'Avg F1 Score']].mean().reset_index()
            mean_df_melted = mean_df.melt(id_vars='Resolution',
                                          value_vars=['Avg Precision', 'Avg Recall', 'Avg F1 Score'],
                                          var_name='Metric', value_name='Score')

            show_legend = (col_idx == 0)

            sns.lineplot(data=mean_df_melted, x='Resolution', y='Score', hue='Metric',
                         palette=palette, marker='o', ax=ax, legend=show_legend)

            if show_legend and ax.get_legend():
                legend = ax.get_legend()
                legend.set_title('Metric')
                legend.get_frame().set_alpha(0.8)
                legend.get_frame().set_boxstyle('round,pad=0.2')
                legend.set_bbox_to_anchor((0.89, 0.5))  # right-middle inside the axes
                legend._loc = 10  # center inside axes

            ax.set_xlabel('Grid Resolution (km)', fontsize=12)
            ax.xaxis.set_major_locator(MultipleLocator(tick))
            ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            if col_idx == 0:
                ax.set_ylabel(f'Threshold: {threshold}\nScore', fontsize=12)

            if row_idx == 0:
                ax.set_title(f'Size: {size}', fontsize=14, pad=20)

            if not show_legend and ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.75,     # 92% of figure height from bottom
        bottom=0.05,  # 5% of figure height from bottom
        left=0.1,     # 10% of figure width from left
        right=0.9,    # 90% of figure width from left
        hspace=0.3,   # 30% of subplot height between rows
        wspace=0.2    # 20% of subplot width between columns
    )

    fig.patch.set_facecolor('#f0f0f0')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('white')

    plt.show()


def plot_layers_vs_resolution(
    df, 
    city, scheme, measure, bucketing_method, 
    thresholds, size, tick,
    ci=None,
    palette = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
]
):
    metrics = ['Avg Precision', 'Avg Recall', 'Avg F1 Score']
    df_long = (
        df[df['Threshold'].isin(thresholds)]
        .melt(
            id_vars=['Resolution', 'Layers', 'Threshold'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
    )

    if df_long.empty:
        print(f"Warning: No data found for the specified thresholds {thresholds}")
        return

    unique_thresholds = df_long['Threshold'].unique()
    unique_metrics = df_long['Metric'].unique()

    fig, axes = plt.subplots(
        len(unique_thresholds),
        len(unique_metrics),
        figsize=(10 * len(unique_metrics), 5 * len(unique_thresholds)),
        sharex=False,  # independent x-axis
        sharey=False
    )

    if len(unique_thresholds) == 1:
        axes = [axes]
    if len(unique_metrics) == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle('Layer Count Effect on Metrics Across Grid Resolutions', fontsize=20, y=0.99)
    plt.figtext(0.5, 0.96, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | '
                           f'Bucketing Method: {bucketing_method.upper()} | Size: {size}',
                ha='center', fontsize=14, style='italic')

    for row_idx, threshold in enumerate(unique_thresholds):
        for col_idx, metric in enumerate(unique_metrics):
            ax = axes[row_idx][col_idx]
            subset = df_long[(df_long['Threshold'] == threshold) & (df_long['Metric'] == metric)]

            show_legend = (col_idx == 0)

            sns.lineplot(
                data=subset,
                x='Resolution',
                y='Score',
                hue='Layers',
                marker='o',
                palette=palette,
                ci=ci,
                ax=ax,
                legend=show_legend
            )

            if show_legend and ax.get_legend():
                legend = ax.get_legend()
                legend.set_title('Layers')
                legend.get_frame().set_alpha(0.8)
                legend.get_frame().set_boxstyle('round,pad=0.2')
                legend.set_bbox_to_anchor((0.89, 0.5))
                legend._loc = 10

            ax.set_xlabel('Grid Resolution (km)', fontsize=12)
            ax.xaxis.set_major_locator(MultipleLocator(tick))
            ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            if col_idx == 0:
                ax.set_ylabel(f'Threshold: {threshold}\nScore', fontsize=12)

            if row_idx == 0:
                ax.set_title(metric.replace('Avg ', ''), fontsize=14, pad=20)

            if not show_legend and ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.92,
        bottom=0.05,
        left=0.1,
        right=0.9,
        hspace=0.3,
        wspace=0.2
    )

    fig.patch.set_facecolor('#f0f0f0')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('white')

    plt.show()
    
def plot_layers_vs_resolution2(
    dfs, 
    data_sizes,
    city, scheme, measure, bucketing_method, 
    tick,
    ci=None,
    palette = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'  # light blue
        
]
):
    metrics = ['Avg Precision', 'Avg Recall', 'Avg F1 Score']
    thresholds = [0.1]
    
    total_rows = len(data_sizes) * len(thresholds)
    total_cols = len(metrics)
    
    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(11 * total_cols, 5 * total_rows),
        sharex=False,
        sharey=False  # Allow different y-axis scales
    )
    
    if total_rows == 1:
        axes = [axes]
    if total_cols == 1:
        axes = [[ax] for ax in axes]
    
    fig.suptitle('Layer Count Effect on Metrics Across Grid Resolutions', fontsize=20, y=0.99)
    plt.figtext(0.5, 0.90, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | '
                           f'Bucketing Method: {bucketing_method.upper()} | Size: {data_sizes[0]}',
                ha='center', fontsize=14, style='italic')
    
    y_labels = {
        'Avg Precision': 'Precision Score',
        'Avg Recall': 'Recall Score',
        'Avg F1 Score': 'F1 Score'
    }
    
    for df_idx, (df, data_size) in enumerate(zip(dfs, data_sizes)):
        for threshold_idx, threshold in enumerate(thresholds):
            row_idx = df_idx * len(thresholds) + threshold_idx
            
            df_long = (
                df[df['Threshold'] == threshold]
                .melt(
                    id_vars=['Resolution', 'Layers', 'Threshold'],
                    value_vars=metrics,
                    var_name='Metric',
                    value_name='Score'
                )
            )
            
            if df_long.empty:
                print(f"Warning: No data found for data size {data_size} with threshold {threshold}")
                continue
            
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx][col_idx]
                subset = df_long[df_long['Metric'] == metric]
                
                sns.lineplot(
                    data=subset,
                    x='Resolution',
                    y='Score',
                    hue='Layers',
                    marker='o',
                    palette=palette,
                    ci=ci,
                    ax=ax,
                    legend=True  # Always show legend
                )
                
                legend = ax.get_legend()
                legend.set_title('Layers')
                legend.get_frame().set_alpha(0.8)
                legend.get_frame().set_boxstyle('round,pad=0.2')
                
                if col_idx == 1:
                    legend.set_bbox_to_anchor((0.98, 0.02))
                    legend._loc = 4
                else:
                    legend.set_bbox_to_anchor((0.98, 0.98))
                    legend._loc = 1
                
                ax.set_xlabel('Grid Resolution (km)', fontsize=12)
                ax.xaxis.set_major_locator(MultipleLocator(tick))
                ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                
                ymin, ymax = ax.get_ylim()
                padding = (ymax - ymin) * 0.05
                ax.set_ylim(ymin - padding, ymax + padding)
                
                if col_idx == 0:
                    ax.set_ylabel(f'Threshold: {threshold}\n{y_labels[metric]}', fontsize=12)
                else:
                    ax.set_ylabel(f"{y_labels[metric]}", fontsize=12)
                
                if row_idx % len(thresholds) == 0 and threshold_idx == 0:
                    ax.set_title(metric.replace('Avg ', ''), fontsize=14, pad=20)
    
    plt.tight_layout()
    
    plt.subplots_adjust(
        top=0.75,      
        bottom=0.08,   
        left=0.15,     
        right=0.9,    
        hspace=0.3,    
        wspace=0.2    
    )
    
    fig.patch.set_facecolor('#f0f0f0')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('white')
    
    plt.show()


def plot_layers_vs_resolution2_1_2_layout(
    dfs, 
    data_sizes,
    city, scheme, measure, bucketing_method, 
    tick,
    ci=None,
    palette = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'  # light blue
    ]
):
    metrics = ['Avg Precision', 'Avg Recall', 'Avg F1 Score']
    thresholds = [0.1]
    
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    ax_top_left = plt.subplot(gs[0, 0])
    ax_top_right = plt.subplot(gs[0, 1])
    ax_bottom_left = plt.subplot(gs[1, 0])
    ax_bottom_right = plt.subplot(gs[1, 1])
    ax_bottom_right.axis('off')

    axes = [ax_top_left, ax_top_right, ax_bottom_left]
    
    fig.suptitle('Layer Count Effect on Metrics Across Grid Resolutions', fontsize=20, y=0.99)
    plt.figtext(0.5, 0.94, f'City: {city.title()} | Measure: {measure.upper()} | Scheme: {scheme.upper()} | '
                               f'Bucketing Method: {bucketing_method.upper()} | Size: {data_sizes[0]}',
                ha='center', fontsize=14, style='italic')
    
    y_labels = {
        'Avg Precision': 'Precision Score',
        'Avg Recall': 'Recall Score',
        'Avg F1 Score': 'F1 Score'
    }
    
    plot_idx = 0
    for df_idx, (df, data_size) in enumerate(zip(dfs, data_sizes)):
        for threshold_idx, threshold in enumerate(thresholds):
            df_long = (
                df[df['Threshold'] == threshold]
                .melt(
                    id_vars=['Resolution', 'Layers', 'Threshold'],
                    value_vars=metrics,
                    var_name='Metric',
                    value_name='Score'
                )
            )
            
            if df_long.empty:
                print(f"Warning: No data found for data size {data_size} with threshold {threshold}")
                continue
            
            for metric in metrics:
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                subset = df_long[df_long['Metric'] == metric]
                
                sns.lineplot(
                    data=subset,
                    x='Resolution',
                    y='Score',
                    hue='Layers',
                    marker='o',
                    palette=palette,
                    ci=ci,
                    ax=ax,
                    legend=True
                )
                
                legend = ax.get_legend()
                legend.set_title('Layers')
                legend.get_frame().set_alpha(0.8)
                legend.get_frame().set_boxstyle('round,pad=0.2')
                
                if plot_idx == 0:
                    legend.set_bbox_to_anchor((0.99, 0.99))
                    legend._loc = 1
                elif plot_idx == 1:
                    legend.set_bbox_to_anchor((0.99, 0.005))
                    legend._loc = 4
                else:  # Bottom plot
                    legend.set_bbox_to_anchor((0.99, 0.99))
                    legend._loc = 1
                
                ax.set_xlabel('Grid Resolution (km)', fontsize=12)
                ax.xaxis.set_major_locator(MultipleLocator(tick))
                ax.xaxis.set_minor_locator(MultipleLocator(tick / 2))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                
                ymin, ymax = ax.get_ylim()
                padding = (ymax - ymin) * 0.05
                ax.set_ylim(ymin - padding, ymax + padding)
                
                if plot_idx == 0:
                    ax.set_ylabel(f'Threshold: {threshold}\n{y_labels[metric]}', fontsize=12)
                elif plot_idx == 2:
                    ax.set_ylabel(f'Threshold: {threshold}\n{y_labels[metric]}', fontsize=12)
                else:
                    ax.set_ylabel(y_labels[metric], fontsize=12)
                
                # Add title
                ax.set_title(f"{metric.replace('Avg ', '')}", fontsize=14, pad=20)
                
                plot_idx += 1
    
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
        if ax.has_data():
            ax.set_facecolor('white')
    
    plt.show()
 