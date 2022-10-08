import math
import time
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt


# 读取文件,返回读取的训练集和测试集文件
def open_file():
    with open('train-images.idx3-ubyte', 'rb') as train_image:
        train_image_file = train_image.read()
    with open('t10k-images.idx3-ubyte', 'rb') as test_image:
        test_image_file = test_image.read()
    with open('train-labels.idx1-ubyte', 'rb') as train_label:
        train_label_file = train_label.read()
    with open('t10k-labels.idx1-ubyte', 'rb') as test_label:
        test_label_file = test_label.read()
    file_cache = [train_image_file, test_image_file, train_label_file, test_label_file]
    return file_cache


# 读入训练集并得到相应信息，返回训练集数据信息的cache
def get_train_info(file_cache):
    train_img_number = int(file_cache[0][4:8].hex(), 16)
    h_train_image = int(file_cache[0][8:12].hex(), 16)
    w_train_image = int(file_cache[0][12:16].hex(), 16)
    train_image_size = h_train_image * w_train_image
    train_label_number = int(file_cache[2][4:8].hex(), 16)
    train_cache = [train_img_number, train_image_size, train_label_number]
    return train_cache


# 读入测试集并得到相应信息，返回测试集数据信息的cache
def get_test_info(file_cache):
    test_img_number = int(file_cache[1][4:8].hex(), 16)
    h_test_image = int(file_cache[1][8:12].hex(), 16)
    w_test_image = int(file_cache[1][12:16].hex(), 16)
    test_image_size = h_test_image * w_test_image
    test_label_number = int(file_cache[3][4:8].hex(), 16)
    test_cache = [test_img_number, test_image_size, test_label_number]
    return test_cache


# 将训练集图片信息以一每一份图片切分，返回切分后训练集图像的列表(一维)
def sort_train_image(train_cache, file_cache):
    i = 16
    train_images = []
    for j in range(train_cache[0]):  # train_cache[0]
        image = [item for item in file_cache[0][i:i + train_cache[1]]]
        i = i + train_cache[1]
        train_images.append(image)
    return train_images


# 二值化训练集图片
def binarize_train_image(train_cache, file_cache):
    i = 16
    binarize_train_images = []
    for j in range(train_cache[0]):
        image = [item for item in file_cache[0][i:i + train_cache[1]]]
        i = i + train_cache[1]
        for s in range(train_cache[1]):
            if image[s] > 0:
                image[s] = 1
        binarize_train_images.append(image)
    return binarize_train_images


# 将训练集标签信息以一每一份图片切分，返回切分后训练集标签的列表(一维)
def sort_train_label(train_cache, file_cache):
    i = 8
    train_labels = []
    for j in range(train_cache[2]):  # train_cache[2]
        label = file_cache[2][i + j]
        train_labels.append(label)
    return train_labels


# 将测试集以一每一份图片切分，返回切分后测试集的列表(一维)
def sort_test_image(test_cache, file_cache):
    i = 16
    test_images = []
    for n in range(test_cache[0]):  # test_cache[0]
        image = [item for item in file_cache[1][i:i + test_cache[1]]]
        i = i + test_cache[1]
        test_images.append(image)
    return test_images


# 二值化测试集图片
def binarize_test_image(test_cache, file_cache):
    i = 16
    binarize_test_images = []
    for n in range(test_cache[0]):
        image = [item for item in file_cache[1][i:i + test_cache[1]]]
        i = i + test_cache[1]
        for s in range(test_cache[1]):
            if image[s] > 0:
                image[s] = 1
        binarize_test_images.append(image)
    return binarize_test_images


# 将训练集标签信息以一每一份图片切分，返回切分后训练集标签的列表(一维)
def sort_test_label(test_cache, file_cache):
    i = 8
    test_labels = []
    for j in range(test_cache[2]):  # test_cache[2]
        label = file_cache[3][i + j]
        test_labels.append(label)
    return test_labels


# 计算欧式距离
def get_dist(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    temp_dist = v1 - v2
    dist = math.sqrt(np.dot(temp_dist.T, temp_dist))
    return dist


# 基于欧氏距离将一张测试集与所有训练集识别
def end_result(train_images, test_img, train_cache):
    label_count = 0  # 记录发生最小欧式距离的训练图像，方便对应其标签
    min_dist = 5000  # 欧式距离最小值
    for i in range(train_cache[0]):  # test_cache[0]
        dist = get_dist(test_img, train_images[i])
        if dist < min_dist:
            min_dist = dist
            label_count = i
            if min_dist < 1400:
                break
    min_label = train_labels[label_count]
    print("匹配标签：" + str(min_label))
    print("欧式距离：" + str(min_dist))
    if min_dist > 2000:
        min_label = -1
        min_dist = -1
    return min_label, min_dist


# 分析基于欧式距离的膜版匹配识别结果
def analyze_os_distance_result(train_cache, test_images, test_labels):
    correct_recognition = 0
    non_recognition = 0
    wrong_recognition = 0
    for i in range(100):  # test_cache[0]
        label, dist = end_result(train_images, test_images[i], train_cache)
        if label == test_labels[i]:
            correct_recognition += 1
        elif label == -1:
            non_recognition += 1
        else:
            wrong_recognition += 1
    correct_accuracy = (correct_recognition / 100)  # test_cache[0]
    wrong_accuracy = (wrong_recognition / 100)
    print("成功识别率：" + str(correct_accuracy))
    print("错误识别率：" + str(wrong_accuracy))
    print("拒绝识别数量：" + str(non_recognition))


# 展示测试图片
def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 将60000张训练集图片分类
def sort_train_images_class(binarize_train_images, train_labels):
    sorted_train_images_classes = [[] for i in range(10)]  # 对0~9依次归类
    index = 0
    for label in train_labels:
        sorted_train_images_classes[label].append(binarize_train_images[index])
        index += 1
    # print(sorted_train_images_classes)
    return sorted_train_images_classes


# 对0~9每一类xj求和
def sum_x_kj(train_labels, binarize_train_images):
    sum_classes = [np.zeros(784, dtype=np.int) for i in range(10)]  # 统计0~9每一类的向量求和结果
    sorted_train_images_classes = sort_train_images_class(binarize_train_images, train_labels)  # 将分类好的图像信息导入
    # 依次将0~9每一类的向量相加
    for i in range(len(sorted_train_images_classes)):  # 0~9
        for j in range(len(sorted_train_images_classes[i])):  # 每一类数字在训练集中的个数
            sum_classes[i] += sorted_train_images_classes[i][j]
    # print(sum_classes)
    return sum_classes


# 将一张测试集与概率矩阵相乘
def multiply_ndarray(list1, list2):
    list1 = np.array(list1)
    mult = 1
    for i in range(len(list1)):
        if list1[i] == 1:
            mult *= list2[i]
        elif list1[i] == 0:
            mult *= (1 - list2[i])
    return mult


# 使用贝叶斯分类器训练
def bayesModelTrain(train_labels, binarize_train_images, binarize_test_images):
    sum_pj_wi_classes = [np.zeros(784, dtype=np.int) for i in range(10)]  # 统计0~9每一类的向量求和结果
    prior_probability_caches = []  # 数字0~9的先验概率(pwi)
    rates = []  # 存储条件概率
    sort_labels = []  # 存放基于贝叶斯分类器识别的标签
    num_class_appearance = dict(Counter(train_labels[:32000]))  # 统计0~9在32000张训练集中的数量
    # print(num_class_appearance)  # 打印0~9在32000张训练集中的个数
    for i in range(10):
        prior_probability = num_class_appearance[i] / 32000  # train_cache[0]
        prior_probability_caches.append(prior_probability)
    # print(prior_probability_caches)  # 打印0~9的先验概率
    sum_classes = sum_x_kj(train_labels[:32000], binarize_train_images[:32000])
    for i in range(len(sum_classes)):
        sum_pj_wi_classes[i] = (sum_classes[i] + 1) / (num_class_appearance[i] + 2)  # 计算p(j)w(i)(概率估计值)
    # print(sum_pj_wi_classes)
    # 计算条件概率
    for i in range(300):
        max_label = 0
        max_rate = 0
        sum_p_wi_p_x_wi = 0  # 计算p(w0)p(X|w0)+p(w1)p(X|w1)+...+p(w9)p(X|w9)
        for j in range(len(sum_pj_wi_classes)):
            rate = np.sum(multiply_ndarray(binarize_test_images[i], sum_pj_wi_classes[j]))  # 计算条件概率
            rates.append(rate)
            sum_p_wi_p_x_wi += rate * prior_probability_caches[j]
        # 计算后验概率
        for k in range(10):  # len(sum_pj_wi_classes)
            p_wi_x = (rates[k] * prior_probability_caches[k]) / sum_p_wi_p_x_wi
            print("测试集" + str(i + 1) + "是" + str(k) + "的概率为:" + str(p_wi_x))
            if max_rate < p_wi_x:  # 筛选出概率最大对应的标签
                max_label = k
                max_rate = p_wi_x
        print("测试集" + str(i + 1) + "被判定为:" + str(max_label))
        sort_labels.append(max_label)
        print('\n')
        rates.clear()
    return sort_labels


# 分析基于贝叶斯分类器的识别结果
def analyze_bys_sort_result(test_labels, sort_labels):
    datas = []  # 记录识别率
    correct_recognition = 0  # 成功识别率
    correct_classes_num = [0 for x in range(10)]  # 统计各类数字成功识别个数
    test_num_class_appearance = dict(Counter(test_labels[:300]))  # 统计0~9在前300张测试集中的数量
    for i in range(300):
        if sort_labels[i] == test_labels[i]:
            correct_classes_num[sort_labels[i]] += 1
            correct_recognition += 1
        i += 1
    correct_accuracy = round((correct_recognition / 300) * 100, 1)
    print("成功识别率：" + str(correct_accuracy) + "%")
    # print(correct_classes_num)
    for i in range(10):
        correct_classes_num_rate = round(correct_classes_num[i] / test_num_class_appearance[i] * 100, 1)
        print(str(i) + "的识别率为：" + str(correct_classes_num_rate) + "%")
        datas.append(correct_classes_num_rate)
    return datas


# 绘制折线图
def draw_map(datas):
    x = range(10)
    y = datas
    plt.plot(x, y, marker='o', lw=2, ls='-', c='k')
    plt.xticks(np.linspace(0, 9, 10, endpoint=True))
    plt.yticks(np.linspace(75, 100, 15, endpoint=True))
    # 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 打印标题及坐标轴名称
    plt.title('识别结果', fontsize=14)
    plt.xlabel('数字类别', fontsize=14)
    plt.ylabel('识别率', fontsize=14)
    plt.show()


if __name__ == '__main__':
    # 基于欧式距离识别手写数字
    file_cache = open_file()
    train_cache = get_train_info(file_cache)
    test_cache = get_test_info(file_cache)
    train_images = sort_train_image(train_cache, file_cache)
    train_labels = sort_train_label(train_cache, file_cache)
    test_images = sort_test_image(test_cache, file_cache)
    test_labels = sort_test_label(test_cache, file_cache)
    old_time = time.time()
    analyze_os_distance_result(train_cache, test_images, test_labels)
    current_time = time.time()
    print("运行时间为：" + str(current_time - old_time) + "s", end='\n')

    # 读取自己手写数字
    test_path = '6.png'
    img = cv2.imread(test_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    # cv_show('img_gray', img_gray)
    img_gray = cv2.resize(img_gray, (28, 28))  # 重新定义测试图片大小
    # cv_show('img_gray', img_gray)
    thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]  # 二值化
    # cv_show('thresh', thresh)
    thresh = np.array(thresh, dtype=np.uint8).reshape(1, 784)[0]
    end_result(train_images, thresh, train_cache)
    print('\n')

    # 基于贝叶斯分类器识别手写数字
    binarize_train_images = binarize_train_image(train_cache, file_cache)
    binarize_test_images = binarize_test_image(test_cache, file_cache)
    sort_labels = bayesModelTrain(train_labels, binarize_train_images, binarize_test_images)
    datas = analyze_bys_sort_result(test_labels, sort_labels)
    draw_map(datas)
