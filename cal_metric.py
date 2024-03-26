import pandas as pd
import math
from tqdm import tqdm
import numpy as np


def calculate_ndcg(label, predict_list, k):
    """
    计算NDCG@K值。

    参数:
    label (str): 标签项目。
    predict_list (list): 推荐列表。
    k (int): 计算NDCG的位置。

    返回:
    float: NDCG@K值。
    """
    # 检查标签项目是否在推荐列表中
    if label in predict_list:
        position = predict_list.index(label) + 1  # 获取标签项目的位置
        # 计算DCG@K
        dcg_at_k = 1 / math.log2(position + 1) if position <= k else 0

        # 计算IDCG@K (在这种情况下，只有一个相关项目，它应该在理想列表的第一位)
        idcg_at_k = 1 / math.log2(1 + 1)

        # 计算NDCG@K
        ndcg_at_k = dcg_at_k / idcg_at_k
        return ndcg_at_k
    else:
        # 如果标签项目不在推荐列表中，NDCG@K为0
        return 0


def calculate_hit_rate(label, predict_list, k):
    """
    计算单个预测记录的命中率。

    参数:
    label (str): 标签项目。
    predict_list (list): 推荐列表。

    返回:
    int: 如果标签项目出现在推荐列表中，则为1，否则为0。
    """
    return 1 if label in predict_list[:k] else 0


def calculate_mrr_single_query(label, predict, k):
    if label in predict[:k]:
        rank = predict.index(label) + 1
        return 1.0 / rank
    return 0


def calculate_repeat_rate(predict , k):
    counts = {}
    array = predict[:k]
    for item in array:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    dup_count = 0
    for count in counts.values():
        if count > 1:
            dup_count += count-1

    return dup_count


def out_count(predict, item_list, k):
    count = 0
    for item in predict[:k]:
        item = item.strip()
        if item not in item_list:
            # print(f'erro item:{item}')
            count += 1
    return count


item_path_dict = {
    'steam': '/home/wangshuo/codes/scope-rec/data/dataset/steam/items.pickle',
    'movies': '/home/wangshuo/codes/scope-rec/data/dataset/movies/items.pickle',
    'toys': '/home/wangshuo/codes/scope-rec/data/dataset/toys/items.pickle',
}

# read  data
dataset_name = 'movies'
predict_file_path = '/home/wangshuo/codes/scope-rec/snap/0322-movies-trl/Epoch40_SFT_Result_SFTTestSeqRec_Top10.pickle'
# predict_file_path = '/home/wangshuo/codes/scope-rec/snap/0312-steam-method2/Epoch35_SFT_Result_SFTTestSeqRec_Top10.pickle'
# predict_file_path = '/home/wangshuo/codes/scope-rec/snap/0223-movies-scope-mask/Epoch33_SFT_Result_SFTTestSeqRec_CBS_Top10.pickle'

item_path = item_path_dict[dataset_name]
predict_data = pd.read_pickle(predict_file_path)
item_data = pd.read_pickle(item_path)
item_data = set(item_data)
print(f"训练中的item个数为：{len(item_data)}")

# # 重新提取一下对应的item
# for i in range(len(predict_data)):
#     predict = predict_data[i][1]
#     if len(predict) == 1:
#         predict = predict[0].split(' ')
#         predict = list(map(lambda x: x.split('.')[0].strip() if '.' in x else x.strip(), predict))
#         predict_data[i][1] = predict

# cal metric
topk_list = [1,5, 10]

predict_list = []
repeat_rate_dict = {}
NDCG_dict = {}
HR_dict = {}
MRR_dict = {}
out_count_dict = {}
sum_count = 0
for label, predict in tqdm(predict_data, total=(len(predict_data))):
    sum_count += len(predict[:10])
    label = label[0]
    for k in topk_list:
        if k not in NDCG_dict.keys():
            NDCG_dict[k] = []
            HR_dict[k] = []
            MRR_dict[k] = []
            out_count_dict[k] = []
            repeat_rate_dict[k] = []

        num = min(k, len(predict))
        NDCG_dict[k].append(calculate_ndcg(label, predict[:num], k))
        repeat_rate_dict[k].append((calculate_repeat_rate(predict[:num], k)))
        HR_dict[k].append(calculate_hit_rate(label, predict[:num], k))
        MRR_dict[k].append(calculate_mrr_single_query(label, predict[:num], k))
        out_count_dict[k].append(out_count(predict, item_data, k))

print(f'{len(predict_data)}个用户')
print(f'预测了{sum_count}个item')
for k in topk_list:
    NDCG = np.mean(NDCG_dict[k])
    HR = np.mean(HR_dict[k])
    MRR = np.mean(MRR_dict[k])
    out_sum = np.sum(out_count_dict[k])
    repeat_rate = np.sum(repeat_rate_dict[k])
    print(f'NDCG@{k}: {NDCG}')
    print(f'HR@{k}: {HR}')
    print(f'MRR@{k}: {MRR}')
    print(f'out_sum@{k}: {out_sum}')
    print(f'repeat_rate@{k}: {repeat_rate}')
