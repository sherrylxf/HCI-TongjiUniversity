# Image Search Engine

> This project is built based on a **five-stage framework**:
> **Image Upload → Feature Extraction → Similar Image Search → Visual Display → User Favorites**,
> allowing users to upload an image, retrieve visually similar results, and add them to a personal favorites folder.

![Web界面展示.gif](src/Web%E7%95%8C%E9%9D%A2%E5%B1%95%E7%A4%BA.gif)

---

## Project Structure

```
server/
├── imagenet/                      # (Optional) Dataset for pretraining or support
├── static/
│   ├── favorites/                 # Generated user favorites
│   ├── images/                    # Raw images (for display or experiments)
│   └── result/                    # Temp directory for search results
├── templates/
│   └── main.html                  # Frontend interface (HTML + JS)
├── uploads/
│   └── dogs_and_cats/
│       ├── Boots/                # Categorized image folders
│       ├── Sandals/
│       ├── Shoes/
│       └── Slippers/
├── image_vectorizer.py           # CNN-based feature extraction script
├── search.py                     # Vector similarity search logic
├── rest-server.py                # Flask RESTful server
├── saved_features_recom.txt      # Cached image feature vectors
├── neighbor_list_recom.pickle    # Optional vector index (KDTree, FAISS etc.)
```

---

## Tech Stack

* **Frontend**: HTML5 + CSS3 + Vanilla JavaScript
* **Backend**: Flask (Python micro-framework)

---

## Core Functional Modules

| Stage              | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| Image Upload       | Supports uploading `.jpg/.png` files with live preview                     |
| Feature Extraction | Extracts features via CNN and stores in `saved_features_recom.txt`         |
| Similarity Search  | Computes Top-K similar images from the feature matrix                      |
| Visualization      | Shows results in a responsive interface, allows user interaction           |
| Favorites System   | Save selected images to `/static/favorites/{folder_name}` with easy access |

---

## Getting Started

1. **Environment Setup**

> We strongly recommend using the [Anaconda Python Distribution](https://www.anaconda.com/products/distribution):

```bash
conda create -n image-search python=3.9
conda activate image-search
pip install flask numpy opencv-python tensorflow
```

2. **Precompute Feature Vectors**

```bash
python image_vectorizer.py
python search.py
```

3. **Start Flask Server**

```bash
python rest-server.py
```

4. **Use the Web Interface**

* Upload an image (`.jpg` / `.png`)
* Select category and result count
* Click **Search**
* View similar images and optionally add them to your Favorites
* Click **Create Folder** to save to a named folder

---

## Notes & Recommendations

* **Anaconda Environment Recommended**
  Use Conda to manage dependencies and isolate Python versions easily.

* **Pre-extracted Features**
  All image features are precomputed and cached. If new images are added, rerun `image_vectorizer.py`.

* **Scalable Category Structure**
  You can freely extend categories by adding subfolders under `uploads/dogs_and_cats/`.

* **Efficient for Small/Mid-scale Datasets**
  Optimized for datasets under 10,000 images. For large-scale search, consider integrating **FAISS** or **Annoy**.

* **Lightweight Frontend**
  No frontend framework required. Fully built with raw HTML/CSS/JS and easy to integrate.

---

## Web UI Preview

![Web界面展示.gif](src/Web%E7%95%8C%E9%9D%A2%E5%B1%95%E7%A4%BA.gif)
---

## License

This project is intended **for academic and educational use only**.
For commercial purposes, please contact the author for licensing.

---
