# Football Analysis Platform

An end-to-end AI-powered football match analysis system built with YOLOv8, ByteTrack, and Flask.

---

## What it does

Upload any football match video and get:

- **Player tracking** — YOLOv8 detection + ByteTrack multi-object tracking with consistent IDs
- **Team assignment** — KMeans clustering on jersey colours, no manual labelling needed
- **Speed & distance** — Real-world metrics in km/h and metres via perspective homography
- **Pass detection** — Automatic pass and interception detection with distances
- **Heatmaps** — Positional density maps per team using Gaussian KDE
- **Formation detection** — Classifies team formations (4-3-3, 4-2-3-1 etc.) with lineup card
- **Match prediction** — Random Forest model trained on 475,000 historical matches
- **Web dashboard** — Flask app with charts, video upload, and PDF report export

---

## Tech Stack

`YOLOv8` `ByteTrack` `OpenCV` `Flask` `Scikit-learn` `Pandas` `NumPy` `ReportLab` `Chart.js`

---

## Setup

```bash
git clone https://github.com/yourusername/football-analysis-platform.git
cd football-analysis-platform

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Place `best.pt` (YOLOv8 model) in `models/` and `matches.csv`, `elo_ratings.csv` in `datasets/`.

```bash
python train_model.py   # train match prediction model once
python app.py           # start web dashboard at localhost:5000
```

---

## Project Structure

```
├── trackers/               # YOLOv8 + ByteTrack integration
├── team_assigner/          # KMeans jersey colour clustering
├── pass_detector.py        # Pass and interception detection
├── heatmap_generator.py    # Positional density heatmaps
├── formation_detector.py   # Formation detection + lineup card
├── pdf_report.py           # PDF match report generation
├── app.py                  # Flask web application
├── main.py                 # Standalone video processing pipeline
├── datasets/               # matches.csv + elo_ratings.csv
├── templates/              # index.html, intelligence.html, upload.html
└── static/                 # style.css, charts.js
```

---



## License

MIT
