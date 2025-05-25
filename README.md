# Image Search Engine  

> 本项目基于五阶段框架（图像上传 → 特征提取 → 相似搜索 → 可视化展示 → 用户收藏）实现了一个图像搜索引擎，用户可上传图像，获取相似结果并加入收藏夹。

![Web界面展示.gif](src/Web%E7%95%8C%E9%9D%A2%E5%B1%95%E7%A4%BA.gif)

---

## 项目结构

```
server/
├── imagenet/                      # 用于预训练或模型支持的数据（如需）
├── static/
│   ├── favorites/                 # 用户收藏夹生成目录
│   ├── images/                    # 存储原始图片（可用于展示）
│   └── result/                    # 搜索结果图片缓存目录
├── templates/
│   └── main.html                  # 前端页面，基于 HTML + JS
├── uploads/
│   └── dogs_and_cats/
│       ├── Boots/                # 分类图像目录
│       ├── Sandals/
│       ├── Shoes/
│       └── Slippers/
├── image_vectorizer.py           # 图像特征提取脚本（使用 CNN 生成向量）
├── search.py                     # 向量相似度搜索模块
├── rest-server.py                # Flask 后端服务接口
├── saved_features_recom.txt      # 所有图像的特征向量缓存
├── neighbor_list_recom.pickle    # 向量索引缓存（如使用 KDTree 或 FAISS）
```

---

##  技术栈

* **前端**：HTML5 + CSS3 + JavaScript（原生）
* **后端**：Flask（Python 微服务框架）

---

## 核心功能模块

| 功能阶段    | 说明                                                          |
| ------- | ----------------------------------------------------------- |
| 图像上传    | 支持 JPG/PNG 图像上传，前端展示预览图                                     |
| 特征提取    | `image_vectorizer.py` 提取图像特征并缓存至 `saved_features_recom.txt` |
| 相似图像搜索  | `search.py` 根据输入图像在特征矩阵中执行 Top-K 相似搜索                       |
| 前端可视化展示 | `main.html` 展示搜索结果，可收藏、删除、重试                                |
| 收藏夹系统   | 收藏图片后自动复制至 `static/favorites/{folder}`，支持命名与展示              |

---

## 使用说明

1. **初始化环境**

```bash
pip install flask numpy opencv-python tensorflow
```

2. **预提取图像特征**

```bash
python image_vectorizer.py
python search.py
```

3. **启动 Flask 服务**

```bash
python rest-server.py
```

4. **界面操作**

   * 上传图片（.jpg/.png）
   * 选择类别和返回数量
   * 点击搜索按钮，展示相似图像
   * 可点击 "Add to Favorites" 收藏图片
   * 点击 "Create Folder" 创建收藏文件夹

---

## 说明与建议

* **推荐使用 Anaconda 环境**：
  建议使用 [Anaconda](https://www.anaconda.com/products/distribution) 管理 Python 环境，可避免依赖冲突并方便包管理：

  ```bash
  conda create -n image-search python=3.9
  conda activate image-search
  pip install flask numpy opencv-python tensorflow
  ```

*  **默认使用静态特征向量文件**：
  所有图像特征已预先提取并保存在 `saved_features_recom.txt` 中，如添加新图像，请重新运行 `image_vectorizer.py`。

* **图像分类目录结构可扩展**：
  默认按 Boots / Sandals / Shoes / Slippers 分类组织，如需新增类别，只需创建对应子文件夹并放入图像即可。

*  **适合中小规模图像搜索实验系统**：
  对 1 万张以内图像效果良好，如需大规模检索可考虑集成 FAISS 或 Annoy 向量库。

* **前端无框架依赖**：
  页面完全使用 HTML/CSS/原生 JS 编写，加载快、部署轻，可直接嵌入 Flask 项目。

---

## Web界面
![Web界面展示.gif](src/Web%E7%95%8C%E9%9D%A2%E5%B1%95%E7%A4%BA.gif)
---


## License

本项目仅用于学习和研究目的。如需商业使用，请联系作者授权。

---
