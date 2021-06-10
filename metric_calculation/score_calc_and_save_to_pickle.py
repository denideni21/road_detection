"""
In this file:
    1. We iterate over all prediction image directories one by one (containing the phrase 'pred_imgs' in their names
    with method -> calculate_all_scores_from_dir()
        1.1 Iterate over all their masks (true and pred) and calculate TP,TN,FP,FN,Recall,Precision,PA,IoU,F1
        1.2 Save all results to a model-level dict which will be exported in the end
            1.2.a STRUCTURE FOR DICT:
               S = {
                  image_i: {'TP': int,
                            'TN': int,
                            'FP': int,
                            'FN': int,
                            'Recall': float,
                            'Precision': float,
                            'PA': float,
                            'F1': float,
                            'IoU': float
                            },
                  image_i+1: {'TP',
                              ...,
                              'IoU'
                              },
                  ....,
                  image_n: {'TP',
                            ...,
                            IoU'
                            },
                  ALL: {'TPs' : [ints],
                        'TNs': [ints],
                        'FPs': [ints],
                        'FNs': ...,
                        'Recalls': [floats],
                        'Precisions': ...,
                        'PAs': ...,
                        'F1s': ...,
                        'IoUs': ...'
                        },
                  STATISTICS: {
                                'TP': {
                                        MEAN: float,
                                        STDEV: float,
                                        VARIANCE: float
                                     },
                                ...,
                                'Recall': {...},
                                ...
                               }
                }
        1.3 After directory S has been created, it is returned to the caller method which adds it to a top-level dict
            object TopLvlDict. This TopLvlDict maps each model-level dicts to their names -> {
                                                                                                'model_name1: S1,
                                                                                                'model_name2: S2,
                                                                                                    ...,
                                                                                              }
    2. After the TopLvlDir contains all model scores, it is time to organise their Mean scores in a dictionary.
        2.1 We add a key 'GLOBAL_MEANS' to TopLvlDict which is mapped to the following dict:
            TopLvlDict = {
              model_name1: S1,
              model_name2: S2,
              ...,
              'GLOBAL_MEANS : {
                                'TP': [(model_name1, mean_TPs), (model_name2, mean_TPs), ...],
                                ...,
                                'IoU': [(model_name1, meanIoU), ...]
                              }
            }
    3. Then, we use that GLOBAL_MEANS dict to compare model scores and calculate percentage difference, as well as the
    p-values associated with that difference.
        3.1 We generate another dictionary inside of TopLvlDict that contains all percentage diffs between all possible
            pairs of models for all metrics.
            Method used -> calculate_percentage_difference_for_all_mean_metrics_from_top_score()
            'PERCENTAGE_DIFFS_MODELWISE': {
                                            'model_1:model_2' : {
                                                                    'TP': difference in percentage,
                                                                    'FP': difference in %,
                                                                    ...,
                                                                },
                                            'model_1:model_3':  {
                                                                    'TP': difference in percentage,
                                                                    'FP': difference in %,
                                                                    ...,
                                                                },
                                            ...,
                                          }
        3.2 We also generate a similar dictionary inside TopLvlDict that contains all p-values for all compared models
            across all metrics. Method used -> check_differences_between_all_models()
            'T_TEST_RESULTS': {
                                'model_1:model_2': {
                                                     'TP': t-test statistic,
                                                     'FP': t-test statistic,
                                                   },
                                'model_1:model_3':  {
                                                     'TP': t-test statistic,
                                                     'FP': t-test statistic,
                                                     ...,
                                                    },
                                            ...,
                              }
    4. The final dict we get is:
        TopLvlDict object = {
                              model_1: S1,
                              model_2: S2,
                              ...,
                              model_n: Sn,
                              'GLOBAL_MEANS': {means},
                              'PERCENTAGE_DIFFS_MODELWISE': {difference percentages},
                              'T_TEST_RESULTS': {t-test statistics}
                            }
    5. We export the dictionary as pickle file to perma storage
    6. We access the saved into the dictionary statistics and generate graphs and calculate p values
"""

import sys
import os
import pandas

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from statistics import mean, variance, stdev
from PIL import Image
from helpers import *
from scipy import stats

plt.rcParams['figure.figsize'] = [15, 7]


def get_iou_score_for_masks(gt_mask_array, pred_mask_array):
    """
    Calculates the IoU score between two masks
    :param gt_mask_array: a ground-truth mask represented as numpy array
    :param pred_mask_array: a predicted mask represented as numpy array
    :return: [int] the IoU score
    """
    intersection = np.logical_and(gt_mask_array, pred_mask_array)
    union = np.logical_or(gt_mask_array, pred_mask_array)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def get_pixel_performance_for_masks(gt_mask_array, pred_mask_array):
    """
    This method calculates the TP,TN,FP,FP,Recall,Precision and F1 score between two masks
    :param gt_mask_array: a ground-truth mask represented as numpy array
    :param pred_mask_array: a predicted mask represented as numpy array
    :return:
    """
    # Reshape array to shape (N, 3) where N is the product of the width multiplied by the height and 3 is the number of
    # channels of each pixel. The first two channels of the reshaped masks are deleted since they contain zeros in all
    # cases. That is because the class labels are contained only in the Blue color identifier -
    #   background is [0,0,0] and road is [0,0,255]
    gt_mask = np.delete(gt_mask_array.reshape(-1, 3), [0, 1], axis=1)
    pred_mask = np.delete(pred_mask_array.reshape(-1, 3), [0, 1], axis=1)

    # Check whether the shapes of the masks match (they have the same number of pixels)
    assert gt_mask.shape == pred_mask.shape, f'The shapes of the images do not match: \
    \n\t{gt_mask.shape} != {pred_mask.shape}\nConsider reshaping.'

    # count unique values - 0's and 255's. This gives us number of road and background pixel predictions
    # gt_elems, c_gt = np.unique(gt_mask, return_counts=True)
    pred_elems, c_pred = np.unique(pred_mask, return_counts=True)

    # gt_counts = [*zip(gt_elems, c_gt)]
    pred_counts = [*zip(pred_elems, c_pred)]

    # The following conditions are used to get the numbers of matching BACKGROUND predictions b/w
    # the expected mask and the predicted one
    condition_true_0 = (gt_mask == 0)  # put True for every pixel that is a 0 in the gt mask and False elsewhere
    condition_predicted_0 = (pred_mask == 0)  # do same as above for predicted mask
    TN = len(np.where(condition_true_0 & condition_predicted_0)[0])  # get the number of matching True values
    FN = abs(TN - pred_counts[0][1])  # calculate the number of False negatives - the wrong Background predictions
    # print("TN indices:", np.where(condition_true_0 & condition_predicted_0)[0])
    # print(f"TN: {TN}, FN: {FN}, Total: {TN + FN}")

    # The same as above but this time for pixel values 255 (ROAD class)
    condition_true_255 = (gt_mask == 255)
    condition_predicted_255 = (pred_mask == 255)
    TP = len(np.where(condition_true_255 & condition_predicted_255)[0])
    FP = abs(TP - pred_counts[1][1])
    # print("TP indices:", np.where(condition_true_255 & condition_predicted_255)[0])
    # print(f"TP: {TP}, FP: {FP}, Total: {TP + FP}")

    # Using the formulas from Chapter 4. in my dissertation
    accuracy_score = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FP)
    precision = TP / (TP + FN)
    F1 = 2 * ((precision * recall) / ((2 * precision) + recall))

    # Create a dict from the metrics for the two masks
    accuracy_stats = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'PA': accuracy_score,
        'F1': F1,
        'Recall': recall,
        'Precision': precision
    }
    # return dict to callee method
    return accuracy_stats


def calculate_all_scores_from_dir(prediction_parent_dir, true_label_dir_path, prediction_mask_dir_path):
    """
    Creates a dictionary model-lvl dict as explained above in the docstring comment at beginning of the file (Step 1)
    :param prediction_parent_dir: a directory where the predicted masks and ground truths of a model are
    :param true_label_dir_path: the ground truths
    :param prediction_mask_dir_path: the predicted masks
    :return: a model-level dict containing the scores of the certain model for all metrics
    """
    gt_image_names = sorted([img for img in os.listdir(os.path.join(prediction_parent_dir, true_label_dir_path))
                             if 'test' in img])   # sort image names for masks (both gt and pred filenames match)
    pred_image_names = gt_image_names

    # initialise model-level dict
    dir_level_dict = {'STATISTICS': {},
                      'ALL': {'TPs': [],
                              'TNs': [],
                              'FPs': [],
                              'FNs': [],
                              'Recalls': [],
                              'Precisions': [],
                              'PAs': [],
                              'F1s': [],
                              'IoUs': []
                              }
                      }  # Model-wise dict -> model1_name : dir_level_dict in the bigger dict for all models

    # For each pair of gt and corresponding pred masks, calculate scores and add to model-level dict
    for pair in zip(gt_image_names, pred_image_names):
        image_level_dict = {}  # Image-wise dict -> image_i: image_level_dict in the dir_level_dict
        gt_arr = np.array(Image.open(os.path.join(prediction_parent_dir, true_label_dir_path, pair[0])))
        pred_arr = np.array(Image.open(os.path.join(prediction_parent_dir, prediction_mask_dir_path, pair[1])))
        iou = get_iou_score_for_masks(
            gt_mask_array=gt_arr, pred_mask_array=pred_arr
        )
        pixel_stats = get_pixel_performance_for_masks(
            gt_mask_array=gt_arr, pred_mask_array=pred_arr
        )  # pixel_stats is dict with all metrics - TP,TN,FP,FN,PA,Recall,Precision,F1 <-(key names)
        for metric_name in pixel_stats.keys():
            image_level_dict[metric_name] = pixel_stats[metric_name]
            dir_level_dict['ALL'][metric_name+'s'].append(pixel_stats[metric_name])
        image_level_dict['IoU'] = iou
        dir_level_dict['ALL']['IoUs'].append(iou)

        image_file_name = '_'.join(pair[0].split('_')[1:])
        dir_level_dict[image_file_name] = image_level_dict

    # Statistical scores
    for metric_result_name in dir_level_dict['ALL'].keys():
        current_score_list = dir_level_dict['ALL'][metric_result_name]
        dir_level_dict['STATISTICS'][metric_result_name[:-1]] = {
            'MEAN': mean(current_score_list),
            'STDEV': stdev(current_score_list),
            'VARIANCE': variance(current_score_list)
        }

    return dir_level_dict


def calculate_percentage_difference_for_all_mean_metrics_from_top_score(score_dict):
    """
    Calculate the precentage difference between the best model for a given metric and all remaining models.
    This is DEPRECATED and is not used anywhere in the code. Just dont want to delete my initial efforts ;d..
    """
    print('\nCalculating percentage difference between best model for metric and all others for all metrics')
    mean_stats_dict = score_dict['GLOBAL_MEANS']
    metric_names = list(mean_stats_dict.keys())
    # model_names = [tup[0] for tup in mean_stats_dict[metric_names[0]]]
    # print(metric_names, model_names, sep='\n')
    output_lines = ["Percentage differences between the best network for a given metric and the rest of the models"]
    percentage_differences = {}

    # Iterate over all metrics and calculate difference
    for metric in metric_names:
        model_scores_for_metric = sorted(mean_stats_dict[metric], key=lambda tup: tup[1], reverse=True)
        diffs = [(model_scores_for_metric[0][0], 0.0)]
        output_lines.append(f"\nModel with highest {metric}: {model_scores_for_metric[0][0]}")
        output_lines.append('Percentage differences from best:')
        for score in model_scores_for_metric[1:]:
            percent_diff = (model_scores_for_metric[0][1] - score[1]) / score[1] * 100
            diffs.append((score[0], percent_diff))
            output_lines.append(f'\t{score[0]} - {percent_diff:.2f}%')
        print(diffs)
        percentage_differences[metric] = diffs

    score_dict['PERCENTAGE_DIFF_FROM_TOP'] = percentage_differences
    with open('OUTPUT FILES/percentage_diffs_from_top.txt', 'w') as output_file:
        output_file.writelines('\n'.join(output_lines) + '\n')
    return score_dict


def check_percentage_diff_between_two_models(score_dict, model1, model2, metric):
    """
    simple calculation of percentage difference between two models' scores for a given metric
    :param score_dict: A TopLvlDict - see explanation on top of file
    :param model1: A model name
    :param model2: Second model's name
    :param metric: A metric to compare the models agains
    :return: a percentage difference score
    """
    metric_score1 = score_dict[model1]['STATISTICS'][metric]['MEAN']
    metric_score2 = score_dict[model2]['STATISTICS'][metric]['MEAN']
    percent_diff = (metric_score1 - metric_score2) / metric_score2 * 100
    return percent_diff


def check_difference_significance(score_dict, name_ref_dict, model_1, model_2, metric):
    """
    Calculate the p-value of the difference between two models for a given metric. The related (paired) t-test is used
    implemented in scipy package.
    :param score_dict: a TopLvlDict
    :param name_ref_dict: a naming reference Bidict
    :param model_1: first model
    :param model_2: seconf model
    :param metric: metric to compare agains
    :return: a t-test statistic
    """
    metric_scores1 = score_dict[model_1]['ALL'][metric + 's']
    metric_scores2 = score_dict[model_2]['ALL'][metric + 's']
    l1, l2 = [len(x) for x in (metric_scores1, metric_scores2)]
    if l1 > l2:
        metric_scores1 = metric_scores1[:l2]
    elif l1 < l2:
        metric_scores2 = metric_scores2[:l1]
    statistic = stats.ttest_rel(metric_scores1, metric_scores2)
    print(f"The results from the paired t-test for models {name_ref_dict[model_1]} "
          f"and {name_ref_dict[model_2]} for {metric} metric are:\n\t{statistic}")
    return statistic


def check_differences_between_all_models(score_dict, name_ref_dict):
    """
    Calculate the precentage difference between all models for a given TopLvlDict.
    This is explained in docstring on top of file at step 3.
    We also write all the results to an excel table
    :param name_ref_dict: a Bidict from helpers.py file that maps model names to their predictions' dir names
    :param score_dict: a TopLvlDict that has 'GLOBAL_MEANS' key which maps all models to their
                       mean values based on metrics
    :return: a percentage diff dict with keys of metrics and values of lists.
             Each list contains: [(model1,metric_score),  (model2,metric_score), ...]
    """

    def write_dict_to_excell(dict_to_save, filename, values):
        # create a dict and write it to DataFrame and then to Excel table
        model_wise_comparsion_names = list(dict_to_save.keys())
        metric_names = list(dict_to_save[model_wise_comparsion_names[0]].keys())
        # create a dict of columns for the excel table
        metrics_columns = {
            metric_name: [dict_to_save[compared_models_names][metric_name].pvalue if values == 'p-values'
                          else dict_to_save[compared_models_names][metric_name]
                          for compared_models_names in model_wise_comparsion_names]
            for metric_name in metric_names
        }
        # merge two dicts that represent columns -
        #           the first one is the column with compared model names - [model1 : model2, model1 : model3, ...]
        #           the second dict has columns for each metric ordered according to the order of compared model names
        data = {'Model Names': model_wise_comparsion_names} | metrics_columns
        pretty(data)
        #
        if os.path.exists(filename):
            print('File already exists. Change passed filename to method or move the existing file away')
        else:
            df = pandas.DataFrame.from_dict(data=data)
            df.to_excel(os.path.join("OUTPUT FILES", filename))
        # -----------------------------------------------------------------------------------------------------------------

    results = {}
    model_wise_diffs = {}
    model_names = [name for name in score_dict.keys() if 'pred_imgs' in name]
    for models in it.combinations(model_names, 2):
        model_1, model_2 = models

        # Calculate t-test significance
        results[f"{name_ref_dict[model_1]} : {name_ref_dict[model_2]}"] = {}
        model_wise_diffs[f"{name_ref_dict[model_1]} : {name_ref_dict[model_2]}"] = {}
        for metric in score_dict[model_1]['STATISTICS'].keys():
            diff = check_difference_significance(score_dict, name_ref_dict, model_1, model_2, metric)
            percent_diff = check_percentage_diff_between_two_models(score_dict, model_1, model_2, metric)
            results[f"{name_ref_dict[model_1]} : {name_ref_dict[model_2]}"][metric] = diff
            model_wise_diffs[f"{name_ref_dict[model_1]} : {name_ref_dict[model_2]}"][metric] = percent_diff

    original_stdout = sys.stdout
    output_text_file = 't_test_results.txt'
    sys.stdout = open(os.path.join("OUTPUT FILES", output_text_file), 'w')
    pretty(results)
    sys.stdout.close()

    output_text_file = 'modelwise_percentage_diffs.txt'
    sys.stdout = open(os.path.join("OUTPUT FILES", output_text_file), 'w')
    pretty(model_wise_diffs)
    sys.stdout.close()
    sys.stdout = original_stdout

    # ---------------------------- write an Excel table for model_wise_diffs: -------------------------------------
    #   each row starts with "model1 : model2" and is followed by differences in metrics or p-values for these diffs
    #   the first column is model_names while the rest are metric-wise scores.
    write_dict_to_excell(
        dict_to_save=results,
        filename='model_and_metric_wise_ttest_percentage_diff_significance.xlsx',
        values='p-values'
    )

    write_dict_to_excell(
        dict_to_save=model_wise_diffs,
        filename='modelwise_percentage_diffs.xlsx',
        values='percent diffs'
    )

    pickle.dump(results, open('OUTPUT FILES/t_test_diff_significance.pickle', 'wb'))
    score_dict['PERCENTAGE_DIFFS_MODELWISE'] = model_wise_diffs
    score_dict['T_TEST_RESULTS'] = results
    return score_dict


def iterate_over_all_prediction_dirs(path_to_root='', pickle_save_name='all_scores_dict'):
    """
    This method builds the whole TopLvlDictionary object. It constructs the different sub-dict componenets one by one by
    calling the appropriate methods.
    :param path_to_root: The directory where all directories of all models' predicted images reside.
    :param pickle_save_name: The name to use to save the final dictionary object as a pickle.
    :return: the name of the pickled object that was just saved
    """
    root_contents = os.listdir(path_to_root) if path_to_root else os.listdir()
    directories = [directory_name
                   for directory_name in root_contents
                   if 'pred_imgs' in directory_name]
    top_level_dict = {}
    for i, directory in enumerate(directories):
        print(f"{i + 1}. Iterating over directory: '{directory}'")
        top_level_dict[directory] = calculate_all_scores_from_dir(
            prediction_parent_dir=os.path.join(path_to_root, directory),
            true_label_dir_path='ground_truth_masks',
            prediction_mask_dir_path='predicted_masks'
        )
    print('All calculations have finished.')

    #  Get all stats for all metrics into the format below:
    #     means_dict = {metric_name: [(model1_name, mean_score), ..., (last_model_name, mean_score)],
    #                   another_metric_name: [...],
    #                   ...}
    means_dict = {}
    for model_name in top_level_dict.keys():
        current_means_dict = top_level_dict[model_name]['STATISTICS']
        for metric_name in current_means_dict.keys():
            if metric_name not in means_dict:
                means_dict[metric_name] = [(ref_dict[model_name], current_means_dict[metric_name]['MEAN'])]
            else:
                means_dict[metric_name].append((ref_dict[model_name], current_means_dict[metric_name]['MEAN']))
    pretty(d=means_dict, title='Mean scores for each metric and model')
    top_level_dict['GLOBAL_MEANS'] = means_dict
    top_level_dict = calculate_percentage_difference_for_all_mean_metrics_from_top_score(top_level_dict)

    top_level_dict = check_differences_between_all_models(top_level_dict, ref_dict)

    print('\nCreating pickle...')
    pickle.dump(top_level_dict, open(f'{os.path.join("OUTPUT FILES", pickle_save_name)}.pickle', 'wb'))
    print(f'Pickle saved to [{os.path.join("OUTPUT FILES", pickle_save_name)}.pickle] successfully')
    return f'{pickle_save_name}.pickle'


if __name__ == '__main__':
    pickled_dict_filename = iterate_over_all_prediction_dirs(
        path_to_root='ALL PREDICTED IMAGES', pickle_save_name='score_dict_with_percentage_diffs_and_ttest_results'
    )
