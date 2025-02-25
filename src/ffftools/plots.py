import matplotlib.pylab as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import numpy as np

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
        ax=None
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
            # Convert grayscale to RGB only if needed
            #if bg_img.ndim == 2 or (bg_img.ndim == 3 and bg_img.shape[2] == 1):  
            #    bg_img = np.stack([bg_img] * 3, axis=-1)  # Convert grayscale to RGB
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


