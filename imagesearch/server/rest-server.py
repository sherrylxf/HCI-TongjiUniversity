#!flask/bin/python  # 指定使用 Flask 环境的 Python 解释器启动

################################################################################################################################
# 本文件使用 Flask 实现后端服务接口
# 功能包括：图像上传、调用推荐函数、返回相似图像搜索结果
################################################################################################################################

from flask import Flask, jsonify, request, redirect, render_template  # Flask 框架核心组件
from flask_httpauth import HTTPBasicAuth  # 提供基本身份认证（此处未启用）
from werkzeug.utils import secure_filename  # 用于安全地处理文件名
import os  # 操作系统路径与文件
import shutil  # 文件复制、移动、删除等操作
import numpy as np  # 数值计算库
from search import recommend  # 导入自定义的推荐函数
from tensorflow.python.platform import gfile  # 用于文件系统操作
from datetime import datetime  # 用于生成时间戳
import uuid  # 用于生成唯一 session ID
import base64, urllib.parse  # 用于 URL 编码处理（预留）

UPLOAD_FOLDER = 'uploads'  # 上传图片的临时存储目录
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # 支持的图像格式
FAVORITES_FOLDER = 'static/favorites'  # 收藏夹根目录

app = Flask(__name__, static_url_path="")  # 初始化 Flask 应用
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 配置上传路径
app.config['FAVORITES_FOLDER'] = FAVORITES_FOLDER  # 配置收藏路径

auth = HTTPBasicAuth()  # 初始化认证（未使用）

# 加载图像特征向量文件
extracted_features = np.zeros((10000, 2048), dtype=np.float32)  # 创建空特征矩阵
with open('saved_features_recom.txt') as f:
    for i, line in enumerate(f):  # 每行一张图的特征向量
        extracted_features[i, :] = line.split()  # 将特征值填入矩阵中
print("loaded extracted_features")  # 控制台提示

# 判断文件扩展名是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 上传图像并返回搜索结果
@app.route('/imgUpload', methods=['POST'])
def upload_img():
    print("image upload")
    result_dir = 'static/result'
    if gfile.Exists(result_dir):
        shutil.rmtree(result_dir)  # 清空旧结果目录
    os.makedirs(result_dir)  # 创建新目录

    # 校验上传内容是否合法
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    # 保存上传的图像文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 获取分类与返回数量参数
    category = request.form.get('category', 'all')
    folder_name = request.form.get('folder_name', '').strip()
    # 使用上传图片的文件名（不含扩展名）作为收藏夹名
    base_name = os.path.splitext(filename)[0]
    session_id = secure_filename(base_name + "_favorite")

    try:
        k = int(request.form.get('k', 9))
        k = max(1, min(k, 9))  # 限制在 1~9 范围
    except ValueError:
        k = 9

    # 调用推荐算法
    recommend(filepath, extracted_features, k=k, category=category)
    os.remove(filepath)  # 删除上传图像

    # 准备返回 JSON 结果
    image_path = "/result"
    image_list = [os.path.join(image_path, f) for f in sorted(os.listdir(result_dir)) if not f.startswith('.')]
    images = {f'image{i}': img for i, img in enumerate(image_list[:k])}

    session_fav_dir = os.path.join(app.config['FAVORITES_FOLDER'], session_id)
    os.makedirs(session_fav_dir, exist_ok=True)
    images['favorites_folder'] = session_id  # 返回给前端，作为收藏路径标识

    return jsonify(images)


# 接收收藏请求，将图像从结果目录复制到收藏夹
@app.route("/addFavorite", methods=['POST'])
def add_favorite():
    data = request.get_json()  # 获取 JSON 数据
    folder = data.get("folder")
    image_url = data.get("image")

    if not folder or not image_url:
        return jsonify({"error": "Missing folder or image data."}), 400

    # 解析图像路径，生成源路径和目标路径
    image_filename = os.path.basename(image_url)
    src_path = os.path.join("static", "result", image_filename)
    dst_dir = os.path.join(app.config['FAVORITES_FOLDER'], folder)
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, image_filename)
    try:
        shutil.copyfile(src_path, dst_path)  # 复制文件到收藏夹
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 异常处理

    return jsonify({"success": True, "saved_to": dst_path})  # 成功返回

# 主页路由：加载主界面 HTML 页面
@app.route("/")
def main():
    return render_template("main.html")  # 渲染 main.html 页面

# 启动应用：debug 模式、监听所有地址（0.0.0.0）
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')