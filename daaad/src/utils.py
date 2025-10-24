import os
import csv
import time
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, List, Union
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pytorch_lightning.utilities.apply_func import move_data_to_device

sns.set_theme()
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")

MAX_SWARMPLOT_SAMPLES = 100
FIG_SIZE = (8, 6)
DPI_SHOW = 80
DPI_STORE = 400

cmapGR = LinearSegmentedColormap(
    'GreenRed',
    {'red':  ((0.0, 0.0, 0.0),   
              (0.5, 1.0, 1.0),
              (1.0, 0.8, 0.8)),
    'green': ((0.0, 0.6, 0.6),
              (0.5, 1.0, 1.0),
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0), 
              (1.0, 0.0, 0.0))
    })
matplotlib.cm.register_cmap('cmapGR', cmapGR)

def density_plot(features:dict, title:str, axis:plt.axis=None, path:str=None, bw_adjust:float=0.3, **kwargs):
    '''
    Creates a seaborn density plot containing all features provided in "features"

    Parameters
    ----------
    features : dict
        Dictionary where the keys represent the names (put into the legend) and the values represent the features (np.array or dataframe)
    title : str
        Title of the plot
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    bw_adjust : float
        Bandwidth of filter for smoothing the density
    '''

    if not axis:
        _, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI_SHOW)
    else:
        ax = axis

    for dname, ddata in features.items():
        mean, std = ddata.mean(), ddata.std()
        sns.kdeplot(ddata.flatten(), bw_adjust=bw_adjust, fill=True, ax=ax, label=dname)
        ax.axvline(x=mean, linestyle='dashed', color=next(ax._get_lines.prop_cycler)['color'], label=dname + ' mean')
        c = next(ax._get_lines.prop_cycler)['color']
        ax.axvline(x=mean - std, linestyle='dotted', color=c, label=dname + ' std')
        ax.axvline(x=mean + std, linestyle='dotted', color=c, label=dname + ' std')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel(title)


    if kwargs.get('xlim'):
        ax.set_xlim(kwargs.get('xlim'))

    if kwargs.get('ylim'):
        ax.set_ylim(kwargs.get('ylim'))

    if kwargs.get('xaxis_format'):
        ax.xaxis.set_major_formatter(FormatStrFormatter(kwargs.get('xaxis_format')))
    if kwargs.get('yaxis_format'):
        ax.yaxis.set_major_formatter(FormatStrFormatter(kwargs.get('yaxis_format')))

    if path and title:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_densityplot'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if not axis:
        plt.show()
        plt.close()


def scatter_plot(name_a:str, data_a:np.array, name_b:str, data_b:np.array, name_hue:str=None, data_hue:np.array=None, 
        title:str=None, plot_diagonal:bool=False, plot_zero:bool=False, axis:plt.axis=None, path:str=None, **kwargs):
    '''
    Creates a seaborn scatter plot and optionally adds visual aids such as diagonals or zero-verticals

    Parameters
    ----------
    name_a, name_b : str
        Names of the two axes
    data_a, data_b : np.array
        Data of the two axes
    name_hue : str
        If available, the name of the value defining the color of the plot
    data_hue : np.array
        If available, the values for coloring the points
    title : str
        The title of the plot
    plot_diagonal : bool
        Whether to plot a dashed line along the diagonal
    plot_zero : bool
        Whether to plot a dashed, vertical line at x = 0
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    '''
    
    if not axis:
        _, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI_SHOW)
    else:
        ax = axis

    show_legend = True

    # if data_hue is provided, create colored scatterplot
    if data_hue is not None:
        unique_values = np.sort(np.unique(data_hue))
        if len(unique_values) == 1:
            sns.scatterplot(
                x=data_a.flatten(), y=data_b.flatten(), hue=data_hue.flatten(), palette=['green'], ax=ax, legend='brief')
        elif len(unique_values) == 2:
            sns.scatterplot(
                x=data_a.flatten(), y=data_b.flatten(), hue=data_hue.flatten(), palette=['green', 'red'], ax=ax, legend='brief')
        elif len(unique_values) == 3:
            sns.scatterplot(
                x=data_a.flatten(), y=data_b.flatten(), hue=data_hue.flatten(), palette=['green', 'yellow', 'red'], ax=ax, legend='brief')
        else:
            sns.scatterplot(x=data_a.flatten(), y=data_b.flatten(), hue=data_hue.flatten(), palette='cmapGR', ax=ax, legend='brief')
    else:
        show_legend = False
        sns.scatterplot(x=data_a.flatten(), y=data_b.flatten(), ax=ax, legend='brief')

    if plot_diagonal:
        ax.set_xlim([np.min([np.min(data_a), np.min(data_b)]), np.max([np.max(data_a), np.max(data_b)])])
        ax.set_ylim([np.min([np.min(data_a), np.min(data_b)]), np.max([np.max(data_a), np.max(data_b)])])
        ax.plot([np.min([np.min(data_a), np.min(data_b)]), np.max([np.max(data_a), np.max(data_b)])], 
                    [np.min([np.min(data_a), np.min(data_b)]), np.max([np.max(data_a), np.max(data_b)])], linestyle='dashed',
                    linewidth=1, label='optimal', alpha=0.6)
        ax.plot([], [], ' ', label=' ')
        ax.plot([], [], ' ', label='$R^2$ = ' + np.array2string(np.corrcoef(data_a.flatten(), data_b.flatten())[0, 1]**2, precision=3))
        ax.plot([], [], ' ', label='MSE = '+ np.array2string(((data_a - data_b)**2).mean(), precision=3))

    if plot_zero:
        ax.set_ylim([-np.max([np.max(np.abs(data_a)), np.max(np.abs(data_b))]), \
                    np.max([np.max(np.abs(data_a)), np.max(np.abs(data_b))])])
        ax.axhline(y = 0, linestyle='dashed', linewidth=1, label='optimal', alpha=0.6)
                    
    if title:
        ax.set_title(title)

    if kwargs.get('xlim'):
        ax.set_xlim(kwargs.get('xlim'))
    if kwargs.get('ylim'):
        ax.set_ylim(kwargs.get('ylim'))
    
    ax.set_xlabel(name_a)
    ax.set_ylabel(name_b)
    
    if kwargs.get('xaxis_format'):
        ax.xaxis.set_major_formatter(FormatStrFormatter(kwargs.get('xaxis_format')))
    if kwargs.get('yaxis_format'):
        ax.yaxis.set_major_formatter(FormatStrFormatter(kwargs.get('yaxis_format')))

    # at = AnchoredText('$R^2$ = ' + np.array2string(np.corrcoef(data_a.flatten(), data_b.flatten())[0, 1]**2, precision=3) +
    #             '\n$\textit\{MSE\}$ = '+ np.array2string(((data_a - data_b)**2).mean(), precision=3), loc='lower right')
    # # at.patch.set_boxstyle('round, pad=0., rounding_size=0.2')
    # ax.add_artist(at)   

    if show_legend:
        ax.legend(title=name_hue if name_hue is not None else '')

    if path and title and name_a and name_b:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_'+name_a+'_'+name_b+'_scatterplot'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if not axis:
        plt.show()
        plt.close()


def bar_plot(features:dict, title:str, axis:plt.axis=None, path:str=None):
    '''
    Creates a seaborn bar plot for all provided features

    Parameters
    ----------
    features : dict
        Dictionary where the keys represent the names (put into the legend) and the values represent the features (np.array or dataframe)
    title : str
        Title of the plot
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    '''

    if axis is None:
        _, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI_SHOW)
    else:
        ax = axis
    
    if len(features) > 1:
        unique_values = np.array([])
        for dname, ddata in features.items():
            unique_values = np.unique(np.concatenate([unique_values, ddata.flatten()]))

        data = []
        for dname, ddata in features.items():
            data.append(np.hstack([unique_values[:, np.newaxis], 
                                np.array([(ddata == d).sum() for d in unique_values])[:, np.newaxis], 
                                np.array([[dname]]*len(unique_values))
                            ]))

        df = pd.DataFrame(np.vstack(data), columns=[title, 'Samples', 'Set'])
        df['Samples'] = df['Samples'].astype(float)
        sns.barplot(x=title, y='Samples', hue='Set', data=df, ax=ax)
        ax.legend()
    else:
        for _, ddata in features.items():
            sns.barplot(x=np.unique(ddata), y=np.array([(ddata == d).sum() for d in np.unique(ddata)]), color='b', ax=ax)
            ax.set_xlabel(title)
            ax.set_ylabel('Samples')
        
    
    for item in ax.get_xticklabels():
        item.set_rotation(45)

    if path and title:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_barplot'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if not axis:
        plt.show()
        plt.close()


def confusion_matrix_plot(name_a:str, data_a:np.array, name_b:str, data_b:np.array, title:str=None, hide_zero_rows:bool=False, hide_zero_columns:bool=False, 
                          axis:plt.axis=None, path:str=None):
    '''
    Creates a confusion matrix plot

    Parameters
    ----------
    name_a, name_b : str
        Names of the two axes
    data_a, data_b : np.array
        Data of the two axes
    title : str
        The title of the plot
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    '''

    if axis is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=DPI_SHOW)
    else:
        ax = axis
    
    data_a, data_b = data_a.astype(str).flatten(), data_b.astype(str).flatten()
    values = np.unique(np.concatenate([data_a, data_b], axis=0))
    cmat = confusion_matrix(data_a, data_b, labels=values)

    if hide_zero_columns:
        column_values = values[cmat.T.sum(axis=1) > 0]
        cmat = cmat.T[cmat.T.sum(axis=1) > 0].T
    if hide_zero_rows:
        row_values = values[cmat.sum(axis=1) > 0]
        cmat = cmat[cmat.sum(axis=1) > 0]
    if not hide_zero_columns and not hide_zero_rows:
        row_values = values
        column_values = values

    sns.heatmap(pd.DataFrame(cmat, row_values, column_values), annot=True, linewidths=.5, fmt='d', cmap='Blues_r', ax=ax)
    ax.set_xlabel(name_b)
    ax.set_ylabel(name_a)
    ax.set_title(title)

    if path and title:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_confusion_matrix'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if not axis:
        plt.show()
        plt.close()


def swarm_plot(features:dict, data_hue:np.array=None, title:str=None, x_label:str='x_values', y_label:str='Features', 
        plot_zero:bool=True, plot_statistics:bool=True, axis:plt.axis=None, path:str=None, **kwargs):
    '''
    Creates a seaborn swarm plot and optionally adds visual aids such as zero-verticals or statistics

    Parameters
    ----------
    features : dict
        Dictionary where the keys represent the names (put on y axis) and the values represent the data (np.array or dataframe) of the features
    data_hue : np.array
        If available, the values for coloring the points
    title : str
        The title of the plot
    x_label, y_label : str
        Names of the axes
    plot_zero : bool
        Whether to plot a dashed, vertical line at x = 0
    plot_statistics : bool
        Whether to show the box plot
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    '''
    
    if axis is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 2 + 0.5*len(features)), dpi=DPI_SHOW)
    else:
        ax = axis

    x_values = np.concatenate(list(features.values()), axis=0).flatten() if len(features) > 1 else np.array(list(features.values())).flatten()
    feature_names = np.concatenate([np.array([n]*len(d)) for n, d in features.items()], axis=0).flatten().astype(str)

    if data_hue is not None:
        hue = np.array([data_hue.flatten()]*len(features)).flatten()

        if data_hue.dtype in ['int', 'uint', 'float', 'float16', 'float32', 'float64']:
            palette = 'flare'
            bins = np.around(np.linspace(hue.min(), hue.max(), 5), 2)
            hue = np.digitize(hue, bins[:-1])
        else:
            palette = 'tab10'

        if len(x_values) > len(features):
            if len(x_values) / len(features) <= MAX_SWARMPLOT_SAMPLES:
                sns.swarmplot(x=x_values.astype(float), y=feature_names, hue=hue, size=4, edgecolor='gray', palette=palette, ax=ax)
            else:
                sns.stripplot(x=x_values.astype(float), y=feature_names, hue=hue, size=4, jitter=0.25, edgecolor='gray', palette=palette, ax=ax)
        else:
            sns.barplot(x=x_values.astype(float), y=feature_names, color='b', ax=ax)
    else:
        if len(x_values) > len(features):
            if len(x_values) / len(features) <= MAX_SWARMPLOT_SAMPLES:
                sns.swarmplot(x=x_values.astype(float), y=feature_names, size=4, edgecolor='gray', palette='flare', ax=ax)
            else:
                sns.stripplot(x=x_values.astype(float), y=feature_names, size=4, edgecolor='gray', jitter=0.25, palette='flare', ax=ax)
        else:
            sns.barplot(x=x_values.astype(float), y=feature_names, color='b', ax=ax)

    if plot_zero:
        ax.axvline(x=0, linewidth=3, c='black')

    if data_hue is not None and data_hue.dtype in ['int', 'uint', 'float'] and len(x_values) > len(features):
        handles, labels = ax.get_legend_handles_labels()
        labels = [str(bins[int(l)]) for l in labels]
        ax.legend(handles, labels, title=title)

    if kwargs.get('xlim'):
        ax.set_xlim(kwargs.get('xlim'))
    if kwargs.get('ylim'):
        ax.set_ylim(kwargs.get('ylim'))

    if kwargs.get('xaxis_format'):
        ax.xaxis.set_major_formatter(FormatStrFormatter(kwargs.get('xaxis_format')))
    if kwargs.get('yaxis_format'):
        ax.yaxis.set_major_formatter(FormatStrFormatter(kwargs.get('yaxis_format')))

    if plot_statistics and len(x_values) > len(features):
        sns.boxplot(x=x_values.astype(float), y=feature_names, whis=np.inf, palette=sns.color_palette("pastel"), boxprops=dict(facecolor=(0,0,0,0)), linewidth=1)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    if path and title:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_swarm_plot'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if axis is None:
        plt.show()
        plt.close()


def imshow_many(images:List[np.array], title:str, subtitles:List[str]=[], row_width:int=3, colorbars:Union[bool, List[bool]]=False, axis:plt.axis=None, path:str=None):
    '''
    Creates a subplot showing a variable number of images.

    Parameters
    ----------
    images : List[np.array]
        List of 2-D numpy arrays (num_images, dim ax. 1, dim ax. 2, num_channels) representing images to be plotted
    title : str
        The title of the plot
    subtitles : List[str]
        List of titles of subplots
        If len(subtitles) == row_width, only top row has titles
        If len(subtitles) == len(images), all subpltos have a title
    row_width: int
        number of images to place in one row before breaking
    colorbars: Union[bool, List[bool]]
        if bool, determines whether to add a colorbar to all images or not
        if List[bool], element i in list determines if colorbars are added to images in row i
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    '''

    if axis is None:
        plt.figure(figsize=(8, 2 + 2 * (len(images) // row_width)), dpi=DPI_SHOW)
    else:
        raise NotImplementedError('Plotting various images onto an existing axis is not yet supported')
        ax = axis
    
    if len(subtitles) == 0:
        subtitles = [""] * len(images)
    elif len(subtitles) == row_width:
        subtitles += [""] * (len(images) - row_width)
    elif len(subtitles) != len(images):
        raise ValueError("Number of subtitles should either be 0, row_width or len(images)")
    
    for i in range(len(images)):
        plt.subplot(len(images) // row_width + 1, row_width, i+1)
        img = plt.imshow(images[i])
        if len(subtitles) > i:
            plt.title(subtitles[i])
        plt.axis('off')
        if (isinstance(colorbars, bool) and colorbars) or (isinstance(colorbars, list) and colorbars[i % row_width]):
            plt.colorbar(img)

    plt.suptitle(title)

    if path and title:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_confusion_matrix'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if not axis:
        plt.show()
        plt.close()


def imscatter(x:np.array, y:np.array, image:np.array, ax:plt.axis=None, zoom=1):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=DPI_SHOW)
   
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

    return artists


def imscatter_many(x_name:str, x_data:np.array, y_name:str, y_data:np.array, images:List[np.array], title:str, zooms:np.array=None, axis:plt.axis=None, path:str=None, **kwargs):
    '''
    Creates a plot where images are placed at given coordinates

    Parameters
    ----------
    images : List[np.array]
        List of 2-D numpy arrays (num_images, dim ax. 1, dim ax. 2, num_channels) representing images to be plotted
    title : str
        The title of the plot
    subtitles : List[str]
        List of titles of subplots
        If len(subtitles) == row_width, only top row has titles
        If len(subtitles) == len(images), all subpltos have a title
    row_width: int
        number of images to place in one row before breaking
    axis : plt.axis
        The axis where the plots should be places. If None, a new axis is created
    path : str
        Where the plots should be saved. If None, plots are not saved
    '''

    if not axis:
        _, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI_SHOW)
    else:
        ax = axis

    for i in range(len(images)):
        imscatter(x_data[i], y_data[i], images[i], ax=ax, zoom=zooms[i] if zooms else 1)

    if title:
        ax.set_title(title)

    if kwargs.get('xlim'):
        ax.set_xlim(kwargs.get('xlim'))
    if kwargs.get('ylim'):
        ax.set_ylim(kwargs.get('ylim'))
    
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
    if kwargs.get('xaxis_format'):
        ax.xaxis.set_major_formatter(FormatStrFormatter(kwargs.get('xaxis_format')))
    if kwargs.get('yaxis_format'):
        ax.yaxis.set_major_formatter(FormatStrFormatter(kwargs.get('yaxis_format')))

    if path and title and x_name and y_name:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, title+'_'+x_name+'_'+y_name+'_imscatter'), bbox_inches='tight', dpi=DPI_STORE)
        plt.close()
    
    if not axis:
        plt.show()
        plt.close()


def clean_string(string:str) -> str:
    return '_'.join(''.join(e.lower() for e in string if e.isalnum() or e.isspace() or e == '_').split(' '))

def append_to_results(path:str, name:str, test_loss:float, losses:dict, args:dict):
    os.makedirs(path, exist_ok=True)

    output = [time.strftime('%d.%m. %H:%M:%S', time.gmtime(time())), path, name, test_loss, str(losses), str(args)]

    try:
        if not os.path.exists('experiments/results.csv'):
            with open('experiments/results.csv', 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'path', 'experiment_name', 'test_loss', 'losses', 'args'])
        with open('experiments/results.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(output)
    except IOError:
        print('I/O error')
    
def rec_concat_dict(data:List[dict]) -> dict:
    if len(data) == 0:
        return data
    if len(data) == 1:
        return data[0]

    conc_dict = {}
    for key in data[0].keys():
        if isinstance(data[0][key], dict):
            conc_dict[key] = rec_concat_dict([batch[key] for batch in data])
        elif isinstance(data[0][key], np.ndarray):
            vals = [batch[key] for batch in data if key in batch]
            if len(vals) > 0:
                conc_dict[key] = np.concatenate(vals) if len(vals) > 1 else vals[0]
        elif torch.is_tensor(data[0][key]):
            vals = [batch[key] for batch in data if key in batch]
            if len(vals) > 0:
                conc_dict[key] = torch.cat([batch[key] for batch in data], dim=0)
        else:
            raise ValueError()
    return conc_dict

def numpy_dict_to_tensor(numpy_dict:Dict[str, np.array], device:int=None) -> Dict[str, torch.Tensor]:
    conc_dict = {}
    for key in numpy_dict.keys():
        if isinstance(numpy_dict[key], dict):
            conc_dict[key] = numpy_dict_to_tensor(numpy_dict[key], device=device)
        elif isinstance(numpy_dict[key], np.ndarray):
            if device is not None:
                tens = move_data_to_device(torch.tensor(numpy_dict[key]).float(), device)
            else:
                tens = torch.from_numpy(numpy_dict[key]).float()
            conc_dict[key] = tens
        elif torch.is_tensor(numpy_dict[key]):
            conc_dict[key] = numpy_dict[key]
            if device is not None:
                conc_dict[key] = move_data_to_device(conc_dict[key], device)
        else:
            raise ValueError('Unknown type found in dictionary: {numpy_dict[key].__class__}. Should be np.array or tensor.')
    return conc_dict


def torch_dict_to_numpy(torch_dict:Dict[str, torch.Tensor]) -> Dict[str, np.array]:
    conc_dict = {}
    for key in torch_dict.keys():
        if isinstance(torch_dict[key], dict):
            conc_dict[key] = torch_dict_to_numpy(torch_dict[key])
        elif torch.is_tensor(torch_dict[key]):
            conc_dict[key] = torch_dict[key].cpu().detach().numpy()
        else:
            conc_dict[key] = torch_dict[key]
    return conc_dict

def sum_join_dicts(dicts:List[dict]) -> dict:
    res = {}
    for d in dicts:
        for k, v in d.items():
            res[k] = res.get(k, 0.) + v
    return res

def mean_join_dicts(dicts:List[dict]) -> dict:
    res = {}
    counts = {}
    for d in dicts:
        for k, v in d.items():
            res[k] = res.get(k, 0.) + v
            counts[k] = counts.get(k, 0.) + 1
    return {k: res[k] / counts[k] for k in res.keys()}

def make_path_unique(path:str) -> str:
    if os.path.exists(path):
        ix = 1
        if path[-1] == '/':
            path = path[:-1]
        alternative_path = path + '_' + str(ix) + '/'
        while os.path.exists(alternative_path):
            ix += 1
            alternative_path = path + '_' + str(ix) + '/'
        path = alternative_path
        os.makedirs(path)
    return path

def batch_call(data:Dict[str, Union[np.array, torch.Tensor]], fn:callable, device:str, batch_size:int=1, **kwargs) -> Dict[str, torch.Tensor]:
    data = numpy_dict_to_tensor(data, device=device)
    
    res = []
    for i in range(len(data[list(data.keys())[0]]) // batch_size + 1):
        if len(data[list(data.keys())[0]][i * batch_size : (i + 1) * batch_size]) > 0:
            res.append(fn(rec_batch_from_tensor_dict(data, i, batch_size), **kwargs))
    
    return rec_concat_dict(res)

def rec_batch_from_tensor_dict(data:dict, batch_ix:int, batch_size:int) -> dict:
    res = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            res[k] = v[batch_ix * batch_size : (batch_ix + 1) * batch_size]
        else:
            res[k] = rec_batch_from_tensor_dict(v, batch_ix, batch_size)
    return res

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
