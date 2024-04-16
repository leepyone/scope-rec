import pandas as pd
import numpy as np
from collections import Counter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def read_data(dataset_name):
    # dataset_name = 'steam'
    sequential_path = '/home/wangshuo/codes/scope-rec/data/dataset/{}/sequential.pickle'.format(dataset_name)
    metas_path = '/home/wangshuo/codes/scope-rec/data/dataset/{}/metas.pickle'.format(dataset_name)
    save_path = '/home/wangshuo/codes/scope-rec/data/dataset/{}/items_count_dict.pickle'.format(dataset_name)
    test_list = []

    sequentials = pd.read_pickle(sequential_path)
    metas = pd.read_pickle(metas_path)
    item_count_dict = {}
    for user in sequentials.keys():
        sequential = sequentials[user]
        test_list.append(metas[sequential[-1]]['title'])
        for item_id in sequential:
            item_title = metas[item_id]['title']
            if item_title not in item_count_dict.keys():
                item_count_dict[item_title] = 0
            item_count_dict[item_title] +=1

    # pd.to_pickle(item_count_dict, save_path)
    return item_count_dict, test_list

def get_pop_by_percent(item_count_dict, test_list, percent=0.1):
    sorted_count = sorted(item_count_dict.items(), key=lambda x: x[1])
    frequent_len = int(len(sorted_count) * percent)
    array = sorted_count[-frequent_len:-1]
    print(f"\t临界的次数是 {array[0][1]}")
    frequent_list = list(map(lambda x: x[0], array))
    print(f"\t流行物品的个数 {len(frequent_list)}")
    cal_test_pop(frequent_list, test_list)
    return frequent_list

def get_pop_by_average(item_count_dict,test_list):
    counts = np.array(list(item_count_dict.values()))
    mean_count = np.mean(counts)
    sorted_count = sorted(item_count_dict.items(), key=lambda x: x[1])
    frequent_list = []
    for item in sorted_count:
        if item[1] > mean_count:
            frequent_list.append(item[0])
    print(f"\t临界的次数是 {mean_count}")
    print(f"\t流行物品的个数 {len(frequent_list)}")
    cal_test_pop(frequent_list, test_list)
    return frequent_list

def cal_test_pop(frequent_list, test_list):
    frequent_count = 0
    test_count = {}
    for item in test_list:
        if item in frequent_list:
            frequent_count += 1
        if item_count_dict[item] not in test_count.keys():
            test_count[item_count_dict[item]] = 0
        test_count[item_count_dict[item]] += 1
    print(f"\t测试集预测的物品为流行物品的个数为：{frequent_count}")
    # print(test_count)


dataset_name = 'steam'
item_count_dict, test_list = read_data(dataset_name)

# 找到出现最多次数和最少次数的物品
max_count_item = max(item_count_dict, key=item_count_dict.get)
min_count_item = min(item_count_dict, key=item_count_dict.get)
# 计算每种出现次数对应的物品数量
count_frequency = Counter(item_count_dict.values())

counts = np.array(list(item_count_dict.values()))
median_count = np.median(counts)
mean_count = np.mean(counts)
max_count = np.max(counts)
min_count = np.min(counts)

print(f"{dataset_name} 数据集的统计信息如下：\n")

# 根据上面的结果打印信息
print(f"出现次数最多的物品是：{max_count_item}，出现了{item_count_dict[max_count_item]}次。")
print(f"出现次数最少的物品是：{min_count_item}，出现了{item_count_dict[min_count_item]}次。")
# print("每种出现次数对应的物品数量如下：")
# for count, num_items in count_frequency.items():
#     print(f"出现{count}次的物品有{num_items}种。")
print(f"出现次数的中位数是：{median_count}。")
print(f"出现次数的平均数是：{mean_count}。")

print("基于百分比取pop")
get_pop_by_percent(item_count_dict,test_list,0.05)
print("基于平均值取pop")
get_pop_by_average(item_count_dict,test_list)


# sorted_count = sorted(item_count_dict.items(), key=lambda x: x[1])
# pop_size = 0.1
# frequent_len = int(len(sorted_count)*pop_size)
# array = sorted_count[-frequent_len:-1]
# print(array[0])
# frequent_list = list(map(lambda x: x[0], array))
#
# frequent_count = 0
# test_count = {}
# for item in test_list:
#     if item in frequent_list:
#         frequent_count += 1
#     if item_count_dict[item] not in test_count.keys():
#         test_count[item_count_dict[item]] = 0
#     test_count[item_count_dict[item]] += 1
# print(f"测试集预测的物品为流行物品的个数为：{frequent_count}")
# # print(f"测试集预测的物品出现次数为1的个数为：{frequent_one}")
# test_count = sorted(test_count.items(), key=lambda x: x[0])
# # print(len(test_count.keys()))
# print(test_count)

# # 将count_frequency的键和值分别作为x和y轴数据
# x_list = []
# y_list = []
# for item in test_count:
#     x_list.append(item[0])
#     y_list.append(item[1])
#
#
# # plt.figure(figsize=(8, 4))
# plt.plot(x_list, y_list, marker='o', linestyle='-', color='b')
#
# plt.title('Count Frequency of Items')
# plt.xlabel('Count of Occurrences')
# plt.ylabel('Number of Items')
# plt.xticks(x_list)
# plt.grid(True)
#
# # 指定保存路径
# save_path = "/home/wangshuo/codes/scope-rec/data/figures/test_count_frequency_plot.png"
# plt.savefig(save_path)