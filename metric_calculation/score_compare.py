import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas
from adjustText import adjust_text

from helpers import ref_dict, load_pickled_dict, pretty, Bidict


# Matplotlib params
plt.rcParams['figure.figsize'] = [15, 7]
plt.rcParams["font.family"] = "serif"


def plot_scatter_with_annotation_adjustment(x1, y1, ranking, labels, force_points=1,
                                            title='', xlabel='', ylabel='', **kwargs):
    """
    This method creates a scatter plot using the matplotlib interface
    :param x1: the x-axis values
    :param y1: the y-axis values
    :param ranking: a ranking that will be used to colour the images. Usually this is arange(len(x1)).
    :param labels: the names of all points (x,y)
    :param force_points: how far to push each point's label
    :param title: title of the plot
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param kwargs: any additional keyword arguments that might be useful
    :return: the matplotlib pyplot object
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(x1, y1, c=ranking, cmap='gist_rainbow', s=70)

    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    ax.grid(axis='both')
    texts = []
    # Name all points accordingly
    for i, name in enumerate(labels):
        texts.append(plt.text(x1[i], y1[i], name, size=9))
    adjust_text(texts,
                force_points=force_points,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.7), **kwargs
                )
    return plt


def generate_scatter_plot_for_metrics(score_dict, metric1, metric2, score,
                                      name_ref_dict: Bidict, figure_save_path,
                                      save_formats_and_dpi: list, force_points=1,
                                      overwrite_figs=False, **kwargs):
    """
    This is a general method that can generate a scatter plot for any two metrics for all given models. It also saves
    the plots with the given formats.
    :param score_dict: a TopLvlDict explained in score_calc_and_save_to_pickle.py
    :param metric1: metric1 to compare agains
    :param metric2: metric2 for y-axis
    :param score: Mean, Stdev, Variance or any other statistical score of model for the metric
    :param name_ref_dict: a bidirectional dictionary for referencing between model names and
                          their predictions' directory names
    :param figure_save_path: the path where to save the model
    :param save_formats_and_dpi: the formats to save the figures as tuples -> [(format, dpi), (format2, dpi), ...]
    :param force_points: how far to push labels from points (generates a better readable graoh)
    :param overwrite_figs: whether to overwrite the existing figures in the output directory if conflicts occur
    :param kwargs: kwargs...
    :return: None
    """
    stat_dict = score_dict['GLOBAL_MEANS']
    metric1_scores_list = stat_dict[metric1]
    x1 = [x[1] for x in metric1_scores_list]
    y1 = [score_dict[name_ref_dict.inverse[x[0]][0]]['STATISTICS'][metric2][score] for x in metric1_scores_list]
    print(f'Printing x and y axis values:\n {metric1}-> {x1};\n {metric2}-> {y1}.')
    labels = [x[0] for x in metric1_scores_list]
    ranking = np.arange(len(x1)).astype('float32')

    plot_scatter_with_annotation_adjustment(
        x1, y1,
        ranking=ranking,
        force_points=force_points,
        labels=labels,
        **kwargs
    )
    if not os.path.exists(figure_save_path.split('/')[0]):
        os.makedirs(figure_save_path.split('/')[0])

    if overwrite_figs:
        print('Creating figures and images...')
        for format_, dpi in save_formats_and_dpi:
            plt.savefig(fname=f'{figure_save_path}.{format_}', dpi=dpi)

    plt.show()


def write_scores_to_xlsx(score_dict, name_ref_dict: Bidict, file_name):
    """
    Create an Excel table that summarises all metrics. It ranks all models based on their scores across all metrics.
       "model name | metric1 | metric2 | ...
        model1     |  0.2    |   0.4   | ...
                    ..."

    :param score_dict: a TopLvlDict containing all scores
    :param name_ref_dict: a Bidict for name references
    :param file_name: a name for the Excel table
    :return: the pandas DataFrame object that is used as intermediary table (used to convert Dict to Excel)
    """
    labels = [metric_name for metric_name in score_dict['GLOBAL_MEANS'].keys()
              if metric_name not in ['TP', 'TN', 'FP', 'FN']]
    model_names = [name_ref_dict[name] for name in score_dict.keys() if name in name_ref_dict]

    # A dict comprehension that creates a dictionary that contains table column as keys and column contents as values
    metrics_columns = {
        metric_name: [score_dict[name_ref_dict.inverse[model_name][0]]['STATISTICS'][metric_name]['MEAN']
                      for model_name in model_names]
        for metric_name in labels
    }
    columns = {'Model Names': model_names} | metrics_columns
    pretty(columns)

    if os.path.exists(file_name):
        print('File already exists. Change passed filename to method or move the existing file away')
    else:
        df = pandas.DataFrame.from_dict(data=columns)
        df.to_excel(file_name)
        return df


def get_p_difference(pickle_path, model_1, model_2):
    """
    Retrieve the p-value for two models
    :param pickle_path:
    :param model_1:
    :param model_2:
    :return:
    """
    p_values = pickle.load(open(pickle_path, 'rb'))
    pretty(p_values[f"{model_1} : {model_2}"])


if __name__ == '__main__':
    scores_dict = load_pickled_dict('OUTPUT FILES/score_dict_with_percentage_diffs_and_ttest_results.pickle')

    # -------------------------- Generate different plots for different metrics -----------------------------------
    # IoU vs F1
    print('\nCreating an Mean_IoU_vs_F1 plot...')
    graph_title = 'Mean IoU and F$_1$ scores for each model'
    x_axis_label = 'Mean F$_1$-measure'
    y_axis_label = 'Average IoU score (Jaccard index)'
    generate_scatter_plot_for_metrics(
        score_dict=scores_dict,
        metric1='IoU',
        metric2='F1',
        name_ref_dict=ref_dict,
        figure_save_path='Figures and Graphs/Mean_IoU_vs_F1',
        save_formats_and_dpi=[('pdf', 1200), ('png', 600)],
        score='MEAN',
        force_points=3,
        title=graph_title,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        overwrite_figs=True
    )

    # TP vs TN
    print('\nCreating an Mean_TP_vs_TN plot...')
    graph_title = 'Mean $\it{TP}$ and $\it{TN}$ pixel classifications for each model'
    x_axis_label = 'Average True Positives\n(True predictions of pixels as ROAD class)'
    y_axis_label = 'Average True Negatives\n(True predictions of pixels as BACKGROUND class)'
    generate_scatter_plot_for_metrics(
        score_dict=scores_dict,
        metric1='TP',
        metric2='TN',
        name_ref_dict=ref_dict,
        figure_save_path='Figures and Graphs/Mean_TP_vs_TN',
        save_formats_and_dpi=[('pdf', 1200), ('png', 600)],
        score='MEAN',
        force_points=4,
        title=graph_title,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        overwrite_figs=True
    )

    # FP vs FN
    print('\nCreating an Mean_FP_vs_FN plot...')
    graph_title = 'Mean $\it{FP}$ and $\it{FN}$ pixel classifications for each model'
    x_axis_label = 'Average False Positives\n(Actual ROAD pixels classified as BACKGROUND)'
    y_axis_label = 'Average False Negatives\n(Actual BACKGROUND pixels classified as ROAD)'
    generate_scatter_plot_for_metrics(
        score_dict=scores_dict,
        metric1='FP',
        metric2='FN',
        name_ref_dict=ref_dict,
        figure_save_path='Figures and Graphs/Mean_FP_vs_FN',
        save_formats_and_dpi=[('pdf', 1200), ('png', 600)],
        score='MEAN',
        force_points=3,
        title=graph_title,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        overwrite_figs=True
    )

    # Create an Excel table containing all scores
    print('\nCreating an Excel table from all scores...')
    table = write_scores_to_xlsx(
        score_dict=scores_dict,
        name_ref_dict=ref_dict,
        file_name="Figures and Graphs/Score Table NEW.xlsx"
    )
    if table is not None:
        print('Printing table below:')
        print(table)

    print('\nPrinting difference significance between\nTest Model and U-NET AUG accros all metrics...')
    # Get a p-value for difference between two models from dict
    get_p_difference('OUTPUT FILES/t_test_diff_significance.pickle', 'Test Model', 'U-NET AUG')
