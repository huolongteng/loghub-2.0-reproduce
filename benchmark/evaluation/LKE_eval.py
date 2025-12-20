"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os
import time
import csv
import pandas as pd

sys.path.append('../')

from old_benchmark.LKE_benchmark import benchmark_settings
from logparser import LKE
from evaluation.utils.common import datasets, common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average
from evaluation.utils.template_level_analysis import evaluate_template_level
from evaluation.utils.PA_calculator import calculate_parsing_accuracy
from logparser.utils.evaluator import evaluate


def format_ratio_tag(ratio_value):
    ratio_str = str(ratio_value)
    if '.' in ratio_str:
        ratio_str = ratio_str.replace('.', 'p')
    return ratio_str


datasets_2k = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]

datasets_full = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    # "Spark",
    # "Thunderbird",
    # "BGL",
    "HDFS",
]


if __name__ == "__main__":
    args = common_args()
    data_type = "full" if args.full_data else "2k"
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_LKE_{data_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent
    )

    if args.full_data:
        datasets = datasets_full
    else:
        datasets = datasets_2k
    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        if os.path.exists(os.path.join(output_dir, f"{dataset}_{data_type}.log_structured.csv")):
            parser = None
            print("parseing result exist.")
        else:
            parser = LKE.LogParser
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex'],
                'split_threshold': setting['split_threshold']
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file
        )

    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"LKE_{data_type}_complex={args.complex}_frequent={args.frequent}", args.complex, args.frequent)

    if args.freeze:
        if not (0 < args.ratio < 1):
            raise ValueError("ratio must be between 0 and 1 for frozen evaluation")

        ratio_tag = format_ratio_tag(args.ratio)
        freeze_suffix = f"_freeze_r{ratio_tag}"
        freeze_result_file = result_file.replace('.csv', f'{freeze_suffix}.csv')
        with open(os.path.join(output_dir, freeze_result_file), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Dataset', 'parse_time', 'identified_templates',
                             'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])

        for dataset in datasets:
            setting = benchmark_settings[dataset]
            log_file = setting['log_file'].replace("_2k", f"_{data_type}")
            log_file_basename = os.path.basename(log_file)
            indir = os.path.join(input_dir, os.path.dirname(log_file))
            groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')
            if args.oracle_template_correction:
                groundtruth = os.path.join(indir, log_file_basename + '_structured_corrected.csv')

            parser = LKE.LogParser(
                log_format=setting['log_format'],
                indir=indir,
                outdir=output_dir,
                rex=setting['regex'],
                split_threshold=setting['split_threshold']
            )
            parser.logname = log_file_basename
            headers, regex = parser.generate_logformat_regex(setting['log_format'])
            df_log = parser.log_to_dataframe(os.path.join(indir, log_file_basename), regex, headers, setting['log_format'])
            total_lines = len(df_log)
            if total_lines == 0:
                continue

            train_size = int(total_lines * args.ratio)
            if train_size < 1 or train_size >= total_lines:
                raise ValueError("train_size must be within (0, N) for frozen evaluation")
            test_size = total_lines - train_size

            print(f"{total_lines}, {train_size}, {test_size}")

            parse_start = time.time()
            parser.logname = f"{log_file_basename}_freeze_r{ratio_tag}"
            parser.df_log = df_log.iloc[:train_size].copy()
            parser.wordLL = []
            parser.wordLen = []
            parser.groups = []
            parser.loglineNumPerGroup = []
            parser.wordLenPerGroup = []
            parser.wordOccuOfPosiLLD = []
            parser.loglinesOfGroups = []
            parser.flatLogLineGroups = []
            parser.newGroups = []
            parser.templates = []
            parser.dedup_lines = parser.df_log['Content'].map(parser.preprocess)
            for line in parser.dedup_lines:
                word_seq = line.strip().split()
                parser.wordLen.append(len(word_seq))
                parser.wordLL.append(tuple(word_seq))

            parser.clustering()
            parser.splitting()
            parser.extracting()
            num_templates_before = len(parser.templates)
            print(num_templates_before)

            predictions = []
            for _, line in df_log.iloc[train_size:].iterrows():
                event_id, event_template = parser.match(line['Content'])
                parsed_row = line.copy()
                parsed_row['EventId'] = event_id
                parsed_row['EventTemplate'] = event_template
                predictions.append(parsed_row)

            num_templates_after = len(parser.templates)
            print(num_templates_after)
            assert num_templates_after == num_templates_before, "Frozen parsing modified template count"

            parse_time = time.time() - parse_start
            parsed_df = pd.DataFrame(predictions)
            parsed_df.fillna("", inplace=True)
            freeze_parsed_path = os.path.join(output_dir, f"{log_file_basename}_structured{freeze_suffix}.csv")
            parsed_df.to_csv(freeze_parsed_path, index=False)

            groundtruth_df = pd.read_csv(groundtruth, dtype=str)
            groundtruth_df = groundtruth_df.iloc[train_size:].reset_index(drop=True)
            parsed_df = parsed_df.reset_index(drop=True)

            filter_templates = None
            if args.complex != 0:
                template_file = os.path.join(indir, log_file_basename + '_templates.csv')
                df = pd.read_csv(template_file)
                if args.complex == 1:
                    df = df[df['EventTemplate'].str.count('<*>') == 0]
                if args.complex == 2:
                    df = df[(df['EventTemplate'].str.count('<*>') >= 1) & (df['EventTemplate'].str.count('<*>') <= 4)]
                if args.complex == 3:
                    df = df[df['EventTemplate'].str.count('<*>') >= 5]
                filter_templates = df['EventTemplate'].tolist()
            if args.frequent != 0:
                template_file = os.path.join(indir, log_file_basename + '_templates.csv')
                df = pd.read_csv(template_file)
                df_sorted = df.sort_values('Occurrences')
                if args.frequent > 0:
                    n = int(len(df_sorted) / 100.0 * args.frequent)
                    filter_templates = df_sorted['EventTemplate'].tolist()[:n]
                else:
                    n = len(df_sorted) - int(len(df_sorted) / 100.0 * -args.frequent)
                    filter_templates = df_sorted['EventTemplate'].tolist()[n:]

            if filter_templates != None and len(filter_templates) == 0:
                continue

            GA, FGA = evaluate(groundtruth_df, parsed_df, filter_templates)
            PA = calculate_parsing_accuracy(groundtruth_df, parsed_df, filter_templates)
            tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(dataset, groundtruth_df, parsed_df, filter_templates)

            result = dataset + ',' + \
                     "{:.2f}".format(parse_time) + ',' + \
                     str(tool_templates) + ',' + \
                     str(ground_templates) + ',' + \
                     "{:.3f}".format(GA) + ',' + \
                     "{:.3f}".format(PA) + ',' + \
                     "{:.3f}".format(FGA) + ',' + \
                     "{:.3f}".format(PTA) + ',' + \
                     "{:.3f}".format(RTA) + ',' + \
                     "{:.3f}".format(FTA) + '\n'

            with open(os.path.join(output_dir, freeze_result_file), 'a') as summary_file:
                summary_file.write(result)

        freeze_metric_file = os.path.join(output_dir, freeze_result_file)
        post_average(
            freeze_metric_file,
            f"LKE_{data_type}_complex={args.complex}_frequent={args.frequent}{freeze_suffix}",
            args.complex,
            args.frequent
        )
