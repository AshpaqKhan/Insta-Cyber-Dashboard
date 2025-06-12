# 🛡️ Cybersecurity & 📸 Instagram Influencer Analysis Dashboards

This project features two professional **Streamlit dashboards** integrated into a single app:

 🔍 **Cybersecurity Threat Detection** using anomaly detection & classification
 📊 **Instagram Influencer Analysis & Engagement Prediction**

Built by **Ashpaq Khan Pathan**.


📁 Folder Structure

```plaintext
Internship/
│
├── Main_dashboard.py        # Streamlit launcher
├── Cyber_Module.py          # Cybersecurity analysis dashboard
├── Insta_Module.py          # Instagram influencer dashboard
├── CloudWatch_Traffic_Web_Attack.csv   # Sample web traffic dataset
├── top_insta_influencers_data.csv      # Sample Instagram dataset
├── requirements.txt
└── README.md
```



## 🧠 Cybersecurity Dashboard Features

✔ Uploads network/web traffic logs
✔ Performs anomaly detection using:

* Isolation Forest
* One-Class SVM
* Local Outlier Factor

✔ Random Forest Classification based on `waf_rule`
✔ Dynamic visuals including:

* Protocol vs Port Heatmap
* Suspicious Traffic by Country
* Hourly Threat Trend
* Feature Importance



## 📸 Instagram Dashboard Features

✔ Uploads influencer data (CSV)
✔ Cleans followers, likes & engagement rate
✔ Explores:

* Top Influencer Countries
* Follower vs Engagement trends
* Engagement boxplots
  ✔ Predicts `High Engagement` using Random Forest



## 🚀 How to Run

```bash
git clone https://github.com/AshpaqKhan/Cyber-Insta-Dashboard.git
cd Cyber-Instagram-Dashboard
pip install -r requirements.txt
streamlit run Main_dashboard.py
```



## 📦 Requirements

You can install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
```



## 🧑‍💻 Author

**Ashpaq Khan Pathan**
🔗 GitHub: [AshpaqKhan](https://github.com/AshpaqKhan)


