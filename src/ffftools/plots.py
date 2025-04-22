import matplotlib.pylab as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np
from .utils import make_label
from .decorators import *

def plot_trial_trajectory(
        single_patch_data,    
        linecolor = 'black',
        linestyle = '-',
        marker_column='M_Selection_Role',
        marker_mapping={'target': 'o', 'distractor': 'x'},
        marker_size=16,
        color_mapping={'target': 'green', 'distractor': 'red'},
        mark_start='S',
        mark_end='E',
        mark_items=None,
        text_color='black',
        text_offsets = {'start_end': (0, 0), 'items': (0, 0)},        
        background=None,
        image_path='%s.png', 
        image_alpha=0.5,
        image_extend=None,
        patch_size=None,
        ax=None,
        show=False
    ):
    df = single_patch_data
    x, y = df['M_Selection_X'].values, df['M_Selection_Y'].values
    if ax == None:
        f, ax = plt.subplots(1)  
    if image_extend is None:
        image_extend = [
                    -patch_size[0] / 2, patch_size[0] / 2,
                    -patch_size[1] / 2, patch_size[1] / 2
                ]

    if background:  
        if type(background == "str"):
            bg_img = mpimg.imread(image_path%df[background].values[0]) 
            ax.imshow(
                bg_img,
                alpha=image_alpha,
                extent=image_extend
            )
            
        else:
            ax.set_facecolor(background)  

    ax.plot(x, y, color=linecolor, linestyle=linestyle)

    for marker in df[marker_column].unique():
        x, y = df[df[marker_column]==marker]['M_Selection_X'].values, df[df[marker_column]==marker]['M_Selection_Y'].values
        ax.plot(x, y, marker=marker_mapping[marker], color=color_mapping[marker],
            linestyle="", markersize=marker_size)

    if mark_start:
        offset = text_offsets['start_end']
        ax.annotate(mark_start,  xy=(x[0] + offset[0], y[0] + offset[1]), color=text_color,
                    fontsize=marker_size*0.8, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
    
    if mark_end:
        ax.annotate(mark_end,  xy=(x[-1] + offset[0], y[-1] + offset[1]), color=text_color,
            fontsize=marker_size*0.8, weight='heavy',
            horizontalalignment='center',
            verticalalignment='center')

    
    if mark_items:
        offset = text_offsets['items']
        for i, id in enumerate(df[mark_items].values):
            #ax.text(x[i] + offset[0], y[i] + offset[1], id)
            ax.annotate(id,  xy=(x[i] + offset[0], y[i] + offset[1]), color=text_color,
                fontsize=marker_size*0.8, weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')

    if patch_size:
        ax.set_xlim(-patch_size[0] / 2, patch_size[0] / 2)
        ax.set_ylim(-patch_size[1] / 2, patch_size[1] / 2)
    
    ax.set_aspect('equal')

    if show:
        plt.show()

    return ax



def plot_bars(df, score_column, category_column='M_Condition_Name', agg_func='mean', 
              ref_dicts=None, value_text=False, ax=None, show=False, **kwargs):
    """
    Plots a bar chart for the specified column, averaging over M_Participant_ID, with one bar per M_Condition_Name.
    The function supports different aggregation methods (mean, median, min, max) and includes error bars.
    It also allows plotting multiple reference lines from different dictionaries, and connects the individual observations
    horizontally across the bars.

    Args:
        df (pd.DataFrame): DataFrame with the data to plot.
        score_column (str): The column to aggregate and plot.
        category_column (str): Name of the column that determines the categories to plot.
        agg_func (str, optional): Aggregation function ('mean', 'median', 'min', 'max'). Default is 'mean'.
        ref_dicts (list of dict, optional): List of reference dictionaries with expected values per category.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be drawn on the given axes. Default is None.

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """

    df = df.groupby(['M_Participant_ID', category_column]).mean().reset_index()

    valid_agg_funcs = ['mean', 'median', 'min', 'max']
    if agg_func not in valid_agg_funcs:
        raise ValueError(f"Invalid aggregation function: {agg_func}. Choose from {valid_agg_funcs}.")

    if agg_func == 'mean':
        aggregated = df.groupby(category_column)[score_column].agg(['mean', 'sem'])
    elif agg_func == 'median':
        aggregated = df.groupby(category_column)[score_column].agg(['median', 'sem'])
    elif agg_func == 'min':
        aggregated = df.groupby(category_column)[score_column].agg(['min', 'sem'])
    elif agg_func == 'max':
        aggregated = df.groupby(category_column)[score_column].agg(['max', 'sem'])

    # Create a new figure and axis if no axis is passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot bars with error bars (SEM)
    conditions = aggregated.index
    values = aggregated[agg_func]
    errors = aggregated['sem'] if df['M_Participant_ID'].nunique() >= 2 else None

    ax.bar(conditions, values, yerr=errors, capsize=5, edgecolor='black', **kwargs)

    # Plot individual data points and connect them horizontally
    unique_participants = df['M_Participant_ID'].unique()

    for participant in unique_participants:
        participant_data = df[df['M_Participant_ID'] == participant]
        participant_values = [participant_data[participant_data[category_column] == condition][score_column].values[0] for condition in conditions]
        ax.plot(conditions, participant_values, 'o-', color='black', alpha=0.3)

    # Plot multiple reference lines if provided
    if ref_dicts:
        linestyles = ['dashed', 'dashed', 'dashdot', 'solid']  # Styles for different refs
        colors = ['red', 'blue', 'green', 'purple']  # Different colors for multiple refs
        
        for idx, ref_dict in enumerate(ref_dicts):
            linestyle = linestyles[idx % len(linestyles)]
            color = colors[idx % len(colors)]
            
            for condition, ref_value in ref_dict.items():
                if condition in conditions:
                    x_pos = list(conditions).index(condition)
                    ax.hlines(y=ref_value, xmin=x_pos - 0.4, xmax=x_pos + 0.4, 
                              colors=color, linestyles=linestyle, linewidth=2.5, 
                              label=f'Ref {idx+1}: {condition}')

    # Set labels and title
    ax.set_xlabel(f'{make_label(category_column)}')
    ax.set_ylabel(f'{agg_func.capitalize()} of {make_label(score_column)}')
    ax.set_title(f'{agg_func.capitalize()} of {make_label(score_column)} by {make_label(category_column)}')

    if value_text:
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if show==True:
        plt.show()

    return ax



def plot_correlation(df, score_column1, score_column2, category_column='M_Condition_Name',
                                 agg_func='mean', transform1=None, transform2=None,
                                 title=True, show_zero_lines=False, ax=None, show=False,  **kwargs):
    """
    Plots a scatter plot showing the correlation between the aggregated scores of two specified columns 
    (one score per participant) for each category in the category_column. Each category gets its own subplot.

    Args:
        df (pd.DataFrame): DataFrame with the data to plot.
        score_column1 (str): The first score column to correlate.
        score_column2 (str): The second score column to correlate.
        category_column (str, optional): The column that determines the categories. Default is 'M_Condition_Name'.
        agg_func (str, optional): Aggregation function ('mean', 'median', 'min', 'max'). Default is 'mean'.
        transform1 (function, optional): Transformation function to apply to score_column1 (default: None).
        transform2 (function, optional): Transformation function to apply to score_column2 (default: None).
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be drawn on the given axes. Default is None.

    Returns:
        matplotlib.axes.Axes: The axes wifrom ..decorators import *th the plot.
    """
    
    df = COLUMN_FUNCTION_MAP[score_column1](df)
    df = COLUMN_FUNCTION_MAP[score_column2](df)

  

    valid_agg_funcs = ['mean', 'median', 'min', 'max']
    if agg_func not in valid_agg_funcs:
        raise ValueError(f"Invalid aggregation function: {agg_func}. Choose from {valid_agg_funcs}.")

    if transform1:
        df[score_column1] = df[score_column1].apply(transform1)
    if transform2:
        df[score_column2] = df[score_column2].apply(transform2)

    display(df)

    # Aggregate data by participant
    # TODO: This bis is used for the bar plots as well ... extract
    if agg_func == 'mean':
        aggregated = df.groupby(['M_Participant_ID', category_column, 'M_Trial_Index'])[[score_column1, score_column2]].mean().reset_index()
    elif agg_func == 'median':
        aggregated = df.groupby(['M_Participant_ID', category_column, 'M_Trial_Index'])[[score_column1, score_column2]].median().reset_index()
    elif agg_func == 'min':
        aggregated = df.groupby(['M_Participant_ID', category_column, 'M_Trial_Index'])[[score_column1, score_column2]].min().reset_index()
    elif agg_func == 'max':
        aggregated = df.groupby(['M_Participant_ID', category_column, 'M_Trial_Index'])[[score_column1, score_column2]].max().reset_index()

    if ax is None:
        fig, ax = plt.subplots(1, df[category_column].nunique(),
                                figsize=(8 * df[category_column].nunique(), 6),
                                sharey=True)

    if df[category_column].nunique() == 1:
        ax = [ax]


    for i, category in enumerate(df[category_column].unique()):
        current_ax = ax[i]

        category_data = aggregated[aggregated[category_column] == category]
        corr_coef = np.corrcoef(category_data[score_column1], category_data[score_column2])[0, 1]
        current_ax.scatter(category_data[score_column1], category_data[score_column2], **kwargs)

        m, b = np.polyfit(category_data[score_column1], category_data[score_column2], 1)
        current_ax.plot(category_data[score_column1], m * category_data[score_column1] + b, color='red', linestyle='--', linewidth=2)

        current_ax.text(0.05, 0.95, f'Correl: {corr_coef:.2f}', transform=current_ax.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
  
        if show_zero_lines:
            current_ax.axvline(0, linestyle=':', color='black')
            current_ax.axhline(0, linestyle=':', color='black')

        # Set labels and title
        label1 = score_column1
        label2 = score_column2
        
        if transform1 is not None:
            label1 = f"{score_column1} ({transform1.__name__})"
        if transform2 is not None:
            label2 = f"{score_column2} ({transform2.__name__})"
        if title:
            current_ax.set_title(f'Correlation between \n{make_label(score_column1)} and {make_label(score_column2)}\nCategory: {category}')
        if i == 0:
            current_ax.set_ylabel(make_label(label2))
        current_ax.set_xlabel(make_label(label1))
            
    plt.tight_layout()

    if show==True:
        plt.show()

    return ax


import numpy as np
import matplotlib.pyplot as plt

def plot_start_heatmap(df, category_column='M_Condition_Name', bins=(3, 3),
                       cmap='jet', ax=None, show=False):
    """
    Creates a heatmap for each category in the specified category_column, based on the first selection
    of each trial.

    Args:
        fff-compatible df (pd.DataFrame): The DataFrame containing the data.
        category_column (str): The column defining different conditions. Default is 'M_Condition_Name'
        bins (tuple, optional): Number of bins in x and y dimensions (default is (3,3)).
        cmap (str, optional): Colormap to use for the heatmap (default is "Blues").
        ax (matplotlib.axes._subplots.Axes, optional): Axis to plot on. If None, a new figure is created.

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """
    
    # Filter only the first selection of each trial
    df_filtered = df[df['M_Selection_Index'] == 0]

    # Get unique conditions
    conditions = df_filtered[category_column].unique()
    
    # Create figure and axes only if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5), constrained_layout=True)
        if len(conditions) == 1:
            axes = [axes] 

    for ax, condition in zip(axes, conditions):
        # Select data for current condition
        subset = df_filtered[df_filtered[category_column] == condition]
        
        # Compute 2D histogram
        heatmap, x_edges, y_edges = np.histogram2d(subset['M_Selection_X'], subset['M_Selection_Y'], bins=bins)
        
        # Plot heatmap
        im = ax.imshow(heatmap.T, origin="lower", cmap=cmap, aspect="auto",
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

        # Titles and labels
        ax.set_title(f"{condition}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    fig.colorbar(im, ax=ax)

    if show==True:
        plt.show()

    return ax

@requires('C_Selection_Inter-target_Time')
def plot_ITT_development(df, x_axis='C_Selection_Target_Count', ax=None, show=False):
    """
    Plots Inter-target times over either selection index or trial index, etc., averaging over participants.
    Includes SEM error bars.

    Args:
        fff-compatible df (pd.DataFrame): The DataFrame containing the data.
        x_axis (str): The variable to use for the x-axis (e.g., 'M_Selection_Index' or 'M_Trial_Index').
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """

    # Dynamically insert missing columns
    @requires(x_axis)
    def insert_missing_columns(df):
        return df
    df = insert_missing_columns(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Group by participant and condition
    grouped = df.groupby(['M_Participant_ID', 'M_Condition_Name', x_axis])['C_Selection_Inter-target_Time'].mean().reset_index()
    summary = grouped.groupby(['M_Condition_Name', x_axis])['C_Selection_Inter-target_Time'].agg(['mean', 'sem']).reset_index()

    # Plot one line per condition
    for condition, condition_data in summary.groupby('M_Condition_Name'):
        ax.errorbar(
            condition_data[x_axis], condition_data['mean'], yerr=condition_data['sem'], 
            fmt='-o', capsize=5, label=condition
        )

    # Labels and title
    ax.set_xlabel(make_label(x_axis, keep_level=True))
    ax.set_ylabel(make_label('C_Selection_Inter-target_Time (ms)'))
    ax.set_title(f'Inter-target Time over {make_label(x_axis, keep_level=True)}')

    ax.legend(title="Condition")
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def create_trajectory_pdf(
    dataset, output_pdf='output.pdf', 
    grid_size=(16, 6),
    figsize=None,
    **kwargs
    ):
    if figsize == None:
        figsize = (grid_size[1] * (2 * 1.77), grid_size[0] * 2)
    print("Creating PDF with one participant per page. This might take a while ...")
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_pdf) as pdf:
        for p in tqdm(dataset['M_Participant_ID'].unique()):
            dfp = dataset.query("M_Participant_ID == @p")
            f, axs = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)            
            for i, t in enumerate(dfp['M_Trial_Index'].unique()):
                dft = dfp.query("M_Trial_Index == @t")
                ax = axs.flatten()[i]
                plot_trial_trajectory(dft, ax=ax, **kwargs) 
                ax.set_title(dft['M_Condition_Name'].values[0])
                
            plt.title('Participant = ' + str(p))
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        d = pdf.infodict()
        d['Title'] = 'Trial-wise collection trajectory plots'
        d['Author'] = 'ffftools'
