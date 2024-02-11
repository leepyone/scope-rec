import os
import json
import pickle
import random

from Levenshtein import distance
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from openai import OpenAI
import argparse
import pandas as pd
import time

from transformers import StoppingCriteriaList, MaxLengthCriteria

from SFT.SFT_dataloader import Test_task_group_mapping, SFTDataset
from metrics import Metrics
from param import Config
from rewrite_by_gpt import GPT
from utils import *


headers = {"User-Agent": "Test Client"}


def quary_vllm(input_text, args):
    for ii in range(args.try_num):
        pload_search = {
            # "model": args.model_name,
            "prompt": input_text,
            "n": 1,
            "temperature": 0.0,
            "max_tokens": args.gen_max_length,
        }

        pload_sample = {
            # "model": args.model_name,
            "prompt": input_text,
            "n": 1,
            "temperature": 0.7,
            "max_tokens": args.gen_max_length,
            "top_p": 0.2,
            "top_k": 5,
        }
        response = requests.post(f'http://127.0.0.1:{args.port}/generate', headers=headers, json=pload_sample if args.sample else pload_search, stream=False)
        output_data = json.loads(response.content)
        if 'text' not in output_data:
            continue
        output_text = output_data["text"][0][len(input_text):]
        return output_text


def quary_openai(input_text, args):
    for ii in range(args.try_num):
        output = gpt.call(input_text)
        if output is None:
            continue
        return output


def quary_api(d, args):
    global wrongtime
    try:
        if f'{args.model_name}_output' not in d:
            input_text = d['input_text']
            if args.model_name in ['snap/Llama-2-7b-hf-chat/', 'snap/gpt-3.5-turbo-1106/']:
                input_text = d['input_text'].split('\n')
                sub_text1 = input_text[1].strip()
                sub_text2 = input_text[4].split('[/INST]')[0].strip()
                sub_text3 = input_text[4].split('[/INST]')[1].split('[INST]')[1].strip()
                if args.SFT_test_task in ['SFTTestSeqRec', 'SFTTestItemCount']:
                    sub_text4 = 'Notice! Do not explain the reason or include any other words.'
                else:
                    sub_text4 = 'Notice! You need to output category information in the brackets after item titles in this template: 1. title (category). Do not explain the reason or include any other words in the front of titles.'
                input_text = f'{sub_text1} {sub_text2} {sub_text3} \n{sub_text4}'
                d['raw_input_text'] = input_text

            if args.model_name in ['snap/Llama-2-7b-hf-chat/', 'snap/gpt-3.5-turbo-1106/']:
            # if args.model_name in ['snap/gpt-3.5-turbo-1106/']:
                d[f'{args.model_name}_output'] = quary_openai(input_text, args)
            else:
                d[f'{args.model_name}_output'] = quary_vllm(input_text, args)

        assert f'{args.model_name}_output' in d, f'no {args.model_name}_output'
        wrongtime = 0

    except Exception as e:
        print(str(e), flush=True)
        wrongtime += 1
        if wrongtime > 10:
            assert 1 == 0, 'wrong'

    return 1


if __name__ == "__main__":
    def vague_mapping(ts):
        for idx, __ in enumerate(ts):
            if __ in test_data.title2item:
                continue
            for ___ in test_data.title2item:
                if distance(__, ___) <= 3:
                    ts[idx] = ___
                    break

    def process_api_output(d):
        if f'{args.model_name}_output' not in d:
            return d
        if d[f'{args.model_name}_output'] == "":
            d[f'{args.SFT_test_task}_output_title_list'] = []
            return d
        if f'{args.SFT_test_task}_output_title_list' in d:
            return d
        raw_output = d[f'{args.model_name}_output']
        if raw_output[0] == raw_output[-1] == '"' or raw_output[0] == raw_output[-1] == "'":
            raw_output = raw_output[1:-1]

        ts = raw_output.split('\n')
        ts = [rm_idx(_).strip().split('\n')[0].strip() for _ in ts if match_idx(_)]
        if args.SFT_test_task != 'SFTTestSeqRec':
            ts = [re.sub(r' *[(,\[](.*)[),\]]$', '', _) for _ in ts]

        ts = [__[1:-1] if __[0] == __[-1] == "'" or __[0] == __[-1] == "\"" else __ for __ in ts if __ != '']
        ts = [__.strip() for __ in ts]
        ts = ts[:d['input_field_data']['item_count']]

        vague_mapping(ts)
        d[f'{args.SFT_test_task}_output_title_list'] = ts

        return d

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/dataset/sub_movie/', help="processed_data path")
    parser.add_argument('--SFT_test_task', type=str, default='', help='in {SFTTestSeqRec, SFTTestRanking, SFT+TestPersonalControlRec, SFT-TestPersonalControlRec, SFTTestPersonalCategoryRate_xx%, SFTTestItemCount}')
    parser.add_argument("--num_process", type=int, default=80)
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-hf-chat', help="openai model")
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--max_item_length", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=512, help="The max length of input text to gpt")
    parser.add_argument("--gen_max_length", type=int, default=4096)
    parser.add_argument("--candidate_num", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--item_index", type=str, default='title64_t')
    parser.add_argument("--backup_ip", type=str, default='0.0.0.0')
    parser.add_argument("--llama2_chat_template", action='store_true')
    parser.add_argument("--idx", action='store_true')
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--user_control_symbol", action='store_true')
    parser.add_argument("--port", type=int, default=13579)
    args = parser.parse_args()
    args.is_main_process = True
    kwargs = vars(args)
    args = Config(**kwargs)
    print(args)
    gpt = GPT(model_name=args.model_name)
    data = {
        'category': load_pickle(args.data_path + 'category.pickle'),
        'metas': load_pickle(args.data_path + 'metas.pickle'),
        'sequential': load_pickle(args.data_path + 'sequential.pickle'),
        'preference': load_pickle(args.data_path + 'preference.pickle'),
        'intention': load_pickle(args.data_path + 'intention.pickle'),
        'share_chat_gpt': None,
        'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
    }
    TestTaskTemplate = {args.SFT_test_task: Test_task_group_mapping[args.SFT_test_task.split('_')[0]]}
    TestTaskNum = {args.SFT_test_task: 1}
    args.output_path = args.model_name
    if args.model_name in ['snap/Llama-2-7b-hf-chat/', 'snap/gpt-3.5-turbo-1106/']:
        args.output_path = f'{args.model_name}{args.data_path.split("/")[-2]}/'
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
    if args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
        TestSeqRec_Result_file = f'{args.output_path}SFTTestSeqRec_Top{args.topk}_Result.pickle'
        data['SFTTestSeqRec_Result'] = load_pickle(TestSeqRec_Result_file)
    test_data = SFTDataset(args, TestTaskTemplate, TestTaskNum, data, None, 'test')
    metrics_dict = Metrics([args.SFT_test_task], args.topk, test_data.category2item, test_data.title2item)
    result_file = f'{args.output_path}{args.SFT_test_task}_Top{args.topk}_Result{"_Sample" if args.sample else ""}.pickle'

    test_data_list = load_pickle(result_file)
    _test_data_list = [_ for _ in test_data]
    if test_data_list and len(test_data_list) == len(_test_data_list):
        for _, __ in zip(test_data_list, _test_data_list):
            _.update(__)
    else:
        test_data_list = _test_data_list
    # for _ in test_data_list:
    #     if f'{_["task"]}_output_title_list' in _:
    #         _[f'{args.SFT_test_task}_output_title_list'] = _[f'{_["task"]}_output_title_list']
    #         del _[f'{_["task"]}_output_title_list']
    #     if _['task'] == 'SFTTestPersonalCategoryRate':
    #         _['task'] = args.SFT_test_task
    # save_pickle(test_data_list, result_file)

    if args.SFT_test_task in ['SFT+TestPersonalControlRec', 'SFT-TestPersonalControlRec'] or args.SFT_test_task.startswith('SFTTestPersonalCategoryRate'):
        remain_test_data_list = [_ for _, __ in zip(test_data_list, data['SFTTestSeqRec_Result'])
                                 if f'{args.model_name}_output' not in _ and 'SFTTestSeqRec_output_title_list' in __][:]
    else:
        remain_test_data_list = [_ for _ in test_data_list if f'{args.model_name}_output' not in _][:]
    print(f"Loading Test Task: '{args.SFT_test_task}'. Remain Example Count: {len(remain_test_data_list)}")
    print(test_data_list[1]['input_text'])
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        results = list(tqdm(executor.map(lambda d: quary_api(d, args), remain_test_data_list), total=len(remain_test_data_list)))

    if len(remain_test_data_list) > 0:
        save_pickle(test_data_list, result_file)
    if args.model_name not in ['snap/Llama-2-7b-hf-chat/', 'snap/gpt-3.5-turbo-1106/']:
        for step_i, example in tqdm(enumerate(test_data_list)):
            if f'{args.model_name}_output' not in example or f'{args.SFT_test_task}_output_title_list' in example:
                continue
            output_title = example[f'{args.model_name}_output']
            output_title_list = [_.strip() for _ in output_title.strip().split('\n')]
            output_title_list = [rm_idx(_) if args.idx else _ for _ in output_title_list]
            vague_mapping(output_title_list)
            example[f'{args.SFT_test_task}_output_title_list'] = output_title_list
    else:
        with ProcessPoolExecutor(max_workers=args.num_process) as executor:
            results = list(tqdm(executor.map(process_api_output, test_data_list), total=len(test_data_list)))
        test_data_list = results

    for step_i, example in tqdm(enumerate(test_data_list)):
        if f'{args.SFT_test_task}_output_title_list' not in example or len(example[f'{args.SFT_test_task}_output_title_list']) == 0:
            continue
        output_label = [_.strip() for _ in example['output_text'].strip().split('\n')]
        output_label = [rm_idx(_) if args.idx else _ for _ in output_label]
        # example['input_field_data']['SeqRec_Result'] = example['input_field_data']['SeqRec_Result']
        metrics_dict.add_sample(example['task'], example['input_field_data'], example[f'{args.SFT_test_task}_output_title_list'], output_label, vague_mapping=False)

    metrics_dict.print()
    if len(remain_test_data_list) > 0:
        save_pickle(test_data_list, result_file)


