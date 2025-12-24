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
import regex as re

import pandas as pd

sys.path.append('../')

from old_benchmark.IPLoM_benchmark import benchmark_settings
from logparser.IPLoM import LogParser
from evaluation.utils.common import datasets, common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average
from evaluation.utils.template_level_analysis import evaluate_template_level
from evaluation.utils.PA_calculator import calculate_parsing_accuracy
from logparser.utils.evaluator import evaluate

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
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]


def format_ratio_tag(ratio_value):
    ratio_str = str(ratio_value)
    if '.' in ratio_str:
        ratio_str = ratio_str.replace('.', 'p')
    return ratio_str


if __name__ == "__main__":

    args = common_args()
    data_type = "full" if args.full_data else "2k"
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_IPLoM_{data_type}"
    # prepare result_file
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
            parser = LogParser
        # run evaluator for a dataset
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex'],
                'CT': setting['CT'], 'lowerBound': setting['lowerBound']
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"IPLoM_{data_type}_complex={args.complex}_frequent={args.frequent}", args.complex, args.frequent)

    if args.freeze:
        # prepare suffix and summary file for frozen evaluation
        ratio_tag = format_ratio_tag(args.ratio)
        freeze_suffix = f"_freeze_r{ratio_tag}"
        freeze_result_file = result_file.replace('.csv', f'{freeze_suffix}.csv')
        with open(os.path.join(output_dir, freeze_result_file), 'w') as csv_file:
            fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fw.writerow(['Dataset', 'parse_time', 'identified_templates',
                        'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])

        for dataset in datasets:
            setting = benchmark_settings[dataset]
            log_file = setting['log_file'].replace("_2k", f"_{data_type}")
            log_file_basename = os.path.basename(log_file)
            indir = os.path.join(input_dir, os.path.dirname(log_file))
            groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')
            if args.oracle_template_correction:
                groundtruth = os.path.join(indir, log_file_basename + '_structured_corrected.csv')

            # build parser and load dataframe
            parser = LogParser(
                log_format=setting['log_format'],
                indir=indir,
                outdir=output_dir,
                rex=setting['regex'],
                CT=setting['CT'],
                lowerBound=setting['lowerBound']
            )
            parser.logname = log_file_basename

            headers, regex = parser.generate_logformat_regex(setting['log_format'])
            full_log_path = os.path.join(indir, log_file_basename)
            parser.df_log = parser.log_to_dataframe(full_log_path, regex, headers, setting['log_format'])
            df_log = parser.df_log
            total_lines = len(df_log)
            if total_lines == 0:
                continue

            train_size = int(total_lines * args.ratio)
            train_size = min(train_size, total_lines)
            if train_size == total_lines:
                train_size = total_lines - 1
            test_size = total_lines - train_size

            # log train/test split info
            print(f"{total_lines}, {train_size}, {test_size}")

            parse_start = time.time()

            # step 1: feed only training logs into partitions
            for _, line in df_log.iloc[:train_size].iterrows():
                content_line = line['Content']
                if content_line.strip() == "":
                    continue

                if parser.para.rex:
                    for currentRex in parser.para.rex:
                        content_line = re.sub(currentRex, '', content_line)

                tokens = list(filter(lambda x: x != '', re.split(r'[\s=:,]', content_line)))
                if not tokens:
                    tokens = [' ']

                tokens.append(str(line['LineId']))
                target_partition = parser.partitionsL[len(tokens) - 1]
                target_partition.logLL.append(tokens)
                target_partition.numOfLogs += 1

            for partition in parser.partitionsL:
                if partition.numOfLogs == 0:
                    partition.valid = False
                elif parser.para.PST != 0 and 1.0 * partition.numOfLogs / (train_size + 1) < parser.para.PST:
                    for logL in partition.logLL:
                        parser.partitionsL[0].logLL.append(logL)
                        parser.partitionsL[0].numOfLogs += 1
                    partition.valid = False

            # step 2-4: generate templates using only training logs
            parser.Step2()
            parser.Step3()
            parser.Step4()

            num_templates_before = len(parser.eventsL)
            print(num_templates_before)

            predictions = []

            # step 5: match test logs against frozen templates
            for _, line in df_log.iloc[train_size:].iterrows():
                content_line = line['Content']
                if parser.para.rex:
                    for currentRex in parser.para.rex:
                        content_line = re.sub(currentRex, '', content_line)

                tokens = list(filter(lambda x: x != '', re.split(r'[\s=:,]', content_line)))
                if not tokens:
                    tokens = [' ']

                event_id, event_template = parser.match_event(tokens)

                parsed_row = line.copy()
                parsed_row['EventId'] = event_id
                parsed_row['EventTemplate'] = event_template
                if parser.keep_para:
                    parsed_row["ParameterList"] = parser.get_parameter_list(parsed_row)
                predictions.append(parsed_row)

            num_templates_after = len(parser.eventsL)
            print(num_templates_after)

            parse_time = time.time() - parse_start

            parsed_df = pd.DataFrame(predictions)
            parsed_df.fillna("", inplace=True)
            freeze_parsed_path = os.path.join(output_dir, f"{log_file_basename}_structured{freeze_suffix}.csv")
            parsed_df.to_csv(freeze_parsed_path, index=False)

            # prepare ground truth and predictions using only test portion
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
        post_average(freeze_metric_file, f"IPLoM_{data_type}_complex={args.complex}_frequent={args.frequent}{freeze_suffix}", args.complex, args.frequent)
