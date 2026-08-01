# 🛣️ Road Damage Object Detection

Detects and localizes road damage (alligator cracking, block cracking,
potholes, cracks, transverse/longitudinal cracks) in images using a
YOLO-based object detector. Multiple YOLO versions (YOLOv8, YOLOv10, YOLO11)
were trained, tuned, and compared on mAP, precision/recall, IoU, inference
speed, model size, and deployment efficiency — the best-performing model is
deployed here.

## 🔗 Links

- **Training notebook (Kaggle):** [road-damage](https://www.kaggle.com/code/anwernasr/road-damage)

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Download your best-performing YOLO weights (`best.pt`) from the training
notebook's output and place it in this folder before running.

## Project structure

```
road-damage-app/
├── app.py
├── requirements.txt
└── README.md
```

## License

MIT
