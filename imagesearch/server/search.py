################################################################################################################################
# 该文件实现了图像搜索/图像检索功能
# 输入参数包括上传图像路径和已提取的特征向量
################################################################################################################################

import random  # 导入随机数模块
import tensorflow.compat.v1 as tf  # 使用 TensorFlow 1.x 接口
import numpy as np  # 数组运算库
import imageio  # 图像读写库
import os  # 文件系统操作库
import scipy.io  # 处理 .mat 文件等
import time  # 时间模块
from datetime import datetime  # 日期时间处理
from scipy import ndimage  # 图像处理工具
# from scipy.misc import imsave  # 已废弃，用 imageio 替代
imsave = imageio.imsave  # 使用 imageio 的 imsave 保存图片
imread = imageio.imread  # 使用 imageio 的 imread 读取图片
from scipy.spatial.distance import cosine  # 余弦距离函数
# import matplotlib.pyplot as plt  # 可视化模块，未使用
from sklearn.neighbors import NearestNeighbors  # 邻近搜索工具，未使用
import pickle  # 序列化与反序列化
from PIL import Image  # 图像处理库
import gc  # 垃圾回收模块
from tempfile import TemporaryFile  # 创建临时文件的模块
from tensorflow.python.platform import gfile  # TensorFlow 提供的文件接口

# TensorFlow 模型中用到的张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # 约 1.34 亿，用于限制最大图像数

# 获取与输入图像最相似的 top-K 图像
def get_top_k_similar(image_data, pred, pred_final, k):
    print("total data", len(pred))  # 打印数据库中图像数量
    print(image_data.shape)  # 打印输入图像的特征形状

    os.makedirs('static/result', exist_ok=True)  # 创建结果图像目录

    # 计算输入图像与所有图像的余弦距离，并按升序取前 k 个索引
    top_k_ind = np.argsort([cosine(image_data, pred_row) for ith_row, pred_row in enumerate(pred)])[:k]
    print(top_k_ind)  # 打印 top-k 的索引

    # 遍历前 k 个最相似图像，保存到 result 文件夹
    for i, neighbor in enumerate(top_k_ind):
        image = imread(pred_final[neighbor])  # 读取图像
        name = pred_final[neighbor]  # 获取图像原路径
        tokens = name.split("\\")  # 按 Windows 路径分隔符分割
        img_name = tokens[-1]  # 提取文件名
        print(img_name)
        name = 'static/result/' + img_name  # 构造保存路径
        imsave(name, image)  # 保存图像


# 创建 Inception 模型的计算图
def create_inception_graph():
    """从预训练模型文件中创建计算图，返回所需张量"""
    with tf.Session() as sess:
        model_filename = os.path.join('imagenet', 'classify_image_graph_def.pb')  # 模型文件路径
        with gfile.FastGFile(model_filename, 'rb') as f:  # 打开模型文件
            graph_def = tf.GraphDef()  # 定义图结构
            graph_def.ParseFromString(f.read())  # 解析图
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))  # 导入图
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor  # 返回三个关键张量


# 获取图像的 bottleneck 特征
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})  # 运行会话获取特征
    bottleneck_values = np.squeeze(bottleneck_values)  # 去除多余维度
    return bottleneck_values  # 返回特征向量


# 图像推荐主函数，调用模型并执行相似图像检索
def recommend(imagePath, extracted_features, k=9, category='all'):
    tf.reset_default_graph()  # 重置默认图

    config = tf.ConfigProto(device_count={'GPU': 0})  # 禁用 GPU
    sess = tf.Session(config=config)  # 创建会话
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()  # 获取图

    image_data = gfile.FastGFile(imagePath, 'rb').read()  # 读取图像字节
    features = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)  # 提取特征

    # 加载图像路径列表（即检索图像的文件名）
    with open('neighbor_list_recom.pickle', 'rb') as f:
        neighbor_list = pickle.load(f)
    print("loaded images:", len(neighbor_list))

    # 若设置了分类过滤，则筛选包含该分类关键字的图像
    if category != 'all':
        filtered_neighbors = [f for f in neighbor_list if category.lower() in f.lower()]
        if len(filtered_neighbors) < k:
            print(f"[Warning] Not enough images in category '{category}', using fallback to all.")
            filtered_neighbors = neighbor_list  # 回退为全部图像
    else:
        filtered_neighbors = neighbor_list  # 默认使用全部图像

    print(f"Using {len(filtered_neighbors)} images for similarity search...")

    # 提取被筛选图像的特征向量索引并构建子集特征矩阵
    neighbor_indices = [neighbor_list.index(f) for f in filtered_neighbors]
    filtered_features = extracted_features[neighbor_indices]

    # 执行 top-k 检索
    get_top_k_similar(features, filtered_features, filtered_neighbors, k)
