# Mask_R-CNN-
🎭 Mask R-CNN Instance Segmentation | Detects &amp; segments objects in crowded scenes with pixel-perfect masks | ResNet-50-FPN backbone | Pretrained on MS-COCO 90 classes | Fine-tuned on Penn-Fudan | PyTorch + OpenCV | Includes image &amp; video inference 🚀
# 🎭 Mask R-CNN — Instance Segmentation

End-to-end Instance Segmentation using Mask R-CNN in crowded scenes.

## 🧠 Model Details
- **Model**      : Mask R-CNN
- **Backbone**   : ResNet-50 + FPN
- **Pretrained** : MS-COCO 2017 (90 classes)
- **Fine-tuned** : Penn-Fudan Pedestrian Dataset
- **Parameters** : 44,401,393

## 📦 Installation
pip install -r requirements.txt

## 🚀 How to Run

### Run on any image
```python
my_image   = load_image_from_file('your_image.jpg')
my_results = run_inference(model, my_image, score_threshold=0.5)
visualize_results(my_image, my_results)
```

### Run on video
```python
process_video('your_video.mp4', 'output.mp4', score_threshold=0.5)
```

## 📊 What It Detects
- 90 different object classes
- People, cars, animals, furniture and more
- Works on crowded scenes with overlapping objects

## 🎯 Results
- Pixel perfect instance masks
- Bounding boxes per object
- Class labels and confidence scores

## 📁 Project Structure
```
mask-rcnn-instance-segmentation/
   📓 mask_rcnn_jupyter.ipynb   → Complete notebook
   🖼️ output_segmented.png      → Sample output image
   📈 finetuning_loss.png       → Training loss curve
   🎬 output_video_small.mp4    → Sample output video
   📄 requirements.txt          → Dependencies
   📄 README.md                 → This file
```

## 🛠️ Tech Stack
- Python 3.11
- PyTorch 2.1.0
- TorchVision 0.16.0
- OpenCV
- Matplotlib
- MoviePy

## 📚 References
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [PyTorch Documentation](https://pytorch.org)
- [MS-COCO Dataset](https://cocodataset.org)
- [Penn-Fudan Dataset](https://www.cis.upenn.edu/~jshi/ped_html)

## 👤 Author
Kaustubh Gupta

## ⭐ If you found this helpful, please star the repo!
