# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    get_samples_of_custom_split,
    load_gt,
    load_gt_of_sample_tokens,
    load_prediction,
    load_prediction_of_sample_tokens,
)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp, custom_accumulate, calc_md
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.render import class_pr_curve, class_tp_curve, dist_pr_curve, summary_plot, visualize_sample
from nuscenes.utils.splits import is_predefined_split


class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')


        if is_predefined_split(split_name=eval_set):
            self.pred_boxes, self.meta = load_prediction(self.result_path, self.nusc, self.cfg.max_boxes_per_sample, DetectionBox,
                                                        verbose=verbose)
            self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        else:
            sample_tokens_of_custom_split : List[str] = get_samples_of_custom_split(split_name=eval_set, nusc=nusc)
            self.pred_boxes, self.meta = load_prediction_of_sample_tokens(self.result_path, self.cfg.max_boxes_per_sample,
                DetectionBox, sample_tokens=sample_tokens_of_custom_split, verbose=verbose)
            self.gt_boxes = load_gt_of_sample_tokens(nusc, sample_tokens_of_custom_split, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self, custom_evaluate = False) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        import pandas as pd
        # df = pd.DataFrame(columns=['det_name_gt', 'det_name_pred', 'translation_gt', 'translation_pred',
        #                            'rotation_gt', 'rotation_pred', 'size_gt', 'size_pred', 'det_score_pred',
        #                            'sample_token'])
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                if not custom_evaluate:
                    md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                else:
                    md = custom_accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # df.to_csv('/opt/imagry/POC_STREAMPETR/evaluation_boxes_CenterNet.csv')

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
                if custom_evaluate:
                    rec = calc_md(metric_data.recall, self.cfg.min_recall, self.cfg.min_precision)
                    metrics.add_label_rec(class_name, dist_th, rec)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True,
             custom_evaluate=False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate(custom_evaluate)

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        if custom_evaluate:
            print('mPre: %.4f' % (metrics_summary['mean_ap']))
            print('mRec: %.4f' % (metrics_summary['mean_rec']))
            print('mF1: %.4f' % (metrics.compute_mf1()))
        else:
            print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        class_recs = metrics_summary['mean_dist_recs']

        if custom_evaluate:
            print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % (
                'Object Class', 'Prec', 'Rec', 'F1', 'ATE', 'ASE',
                'AOE', 'AVE', 'AAE'))
            self.optim_metrics = pd.DataFrame({'class': [], 'precision': [], 'recall': [], 'f1': []})
            for class_name in class_aps.keys():
                f1 = 2*((class_aps[class_name]*class_recs[class_name])/(class_aps[class_name]+class_recs[class_name]))
                print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                      % (class_name, class_aps[class_name], class_recs[class_name], f1,
                         class_tps[class_name]['trans_err'],
                         class_tps[class_name]['scale_err'],
                         class_tps[class_name]['orient_err'],
                         class_tps[class_name]['vel_err'],
                         class_tps[class_name]['attr_err']))
                self.optim_metrics.loc[len(self.optim_metrics)] = [class_name, class_aps[class_name], class_recs[class_name], f1]

        else:
            print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % (
                    'Object Class', 'AP', 'ATE', 'ASE',
                    'AOE', 'AVE', 'AAE'))
            for class_name in class_aps.keys():
                print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                    % (class_name, class_aps[class_name],
                        class_tps[class_name]['trans_err'],
                        class_tps[class_name]['scale_err'],
                        class_tps[class_name]['orient_err'],
                        class_tps[class_name]['vel_err'],
                        class_tps[class_name]['attr_err']))

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

# class DetectionEval2(DetectionEval):
#     """
#     This is the official nuScenes detection evaluation code.
#     Results are written to the provided output_dir.
#
#     nuScenes uses the following detection metrics:
#     - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
#     - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
#     - nuScenes Detection Score (NDS): The weighted sum of the above.
#
#     Here is an overview of the functions in this method:
#     - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
#     - run: Performs evaluation and dumps the metric data to disk.
#     - render: Renders various plots and dumps to disk.
#
#     We assume that:
#     - Every sample_token is given in the results, although there may be not predictions for that sample.
#
#     Please see https://www.nuscenes.org/object-detection for more details.
#     """
#     def __init__(self,
#                  nusc: NuScenes,
#                  config: DetectionConfig,
#                  result_path: str,
#                  eval_set: str,
#                  output_dir: str = None,
#                  verbose: bool = True):
#         # config.min_precision = 0.0
#         # config.min_recall = 0.0
#         # config.dist_ths = [2.0]
#         super().__init__(nusc, config, result_path, eval_set, output_dir, verbose)

cat2indx_nuscenes = {
'car': 0,
'truck': 1,
'construction_vehicle': 2,
'bus': 3,
'trailer': 4,
'barrier': 5,
'motorcycle': 6,
'bicycle': 7,
'pedestrian': 8,
'traffic_cone': 9,
}

def export_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def filter_predictions(preds, conf_thresholds, cat2indx_nuscenes, args, name = ''):
    with open(preds, 'r') as f:
        json_results = json.load(f)
    results_th = {
        token: [det for det in detections if det['detection_score'] > conf_thresholds[cat2indx_nuscenes[det['detection_name']]]]
        for token, detections in json_results['results'].items()
    }
    filtered_results = {'meta': json_results['meta'], 'results': results_th}
    result_path = os.path.join(os.path.dirname(args.result_path),
                                f'{os.path.splitext(os.path.basename(args.result_path))[0]}_th_filtered_{name}.json')
    export_json(result_path, filtered_results)
    return result_path

def compute_conf_thresh(preds, metric, args, cfg_):
    all_dfs = pd.DataFrame()
    print('Compute evaluation for confidence thresholds')
    result_path_ = preds
    for i in tqdm(np.arange(0.3, 0.6, 0.1)):
        i = round(i, 1)
        conf_thresholds = np.array([i] * 10)
        result_path_ = filter_predictions(result_path_, conf_thresholds, cat2indx_nuscenes, args, str(i))
        nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                                   output_dir=output_dir_, verbose=verbose_)
        nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_, custom_evaluate=args.custom)
        df = nusc_eval.optim_metrics
        df['conf_thresh'] = i
        all_dfs = pd.concat([all_dfs, df], ignore_index=True)
    result = all_dfs.loc[all_dfs.groupby('class')[metric].idxmax()]
    conf_dict = {}
    for cat in cat2indx_nuscenes.keys():
        conf_dict[cat] = result[result['class'] == cat]['conf_thresh'].iloc[0]
    export_json(os.path.join(os.path.dirname(result_path_), 'conf_threshold_optims.json'), conf_dict)
    conf_list = list(conf_dict.values())
    print(f'conf_thresholds:{conf_dict}')
    return conf_list


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--custom', action="store_true",
                        help='Whether to use nuscenes evaluation or custom evaluation.')
    parser.add_argument('--conf_thresholds', nargs='*', type=float,
                        help='Filter predictions with conf thresholds.')
    parser.add_argument('--conf_thresholds_optim', type=str, nargs='?', default=None, const='f1',
                        choices=["precision", "recall", "f1"], help='Use and compute conf threshold')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = 0
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))
    if args.conf_thresholds is None:
        conf_thresholds = False
    elif len(args.conf_thresholds) == 0:
        conf_thresholds = [0.3]*10
    else:
        conf_thresholds = args.conf_thresholds
    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)

    if args.custom:
        cfg_.min_precision = 0.0
        cfg_.min_recall = 0.0
        cfg_.dist_ths = [2.0]
        if conf_thresholds:
            result_path_ = filter_predictions(result_path_, conf_thresholds, cat2indx_nuscenes, args)

        elif args.conf_thresholds_optim:
            conf_list = compute_conf_thresh(result_path_, args.conf_thresholds_optim, args, cfg_)
            result_path_ = filter_predictions(result_path_, conf_list, cat2indx_nuscenes, args)

    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                                   output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_, custom_evaluate=args.custom)
