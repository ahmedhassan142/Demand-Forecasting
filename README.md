# 📦 Supply Chain Demand Forecaster

This Gradio app predicts product demand based on:

- 📅 **Month**
- 📦 **Product**
- 🏬 **Store**

Built with an XGBoost machine learning model trained on inventory data.

---

## 🚀 How to Use

1. **Select the Month** (1 = January, 12 = December)
2. **Choose a Product ID**
3. **Choose a Store ID**
4. ✅ Click **Submit**
5. 📈 View the **Predicted Demand**

---

## 🛠️ Powered By

- `Gradio` – interactive UI
- `XGBoost` – ML model
- `scikit-learn`, `pandas` – data processing
- `Hugging Face Spaces` – cloud deployment

---

## 📁 Files in This Space

- `app.py` – Gradio frontend + prediction logic
- `model.pkl` – Trained demand forecasting model
- `product_encoder.pkl` – Product ID encoding
- `store_encoder.pkl` – Store ID encoding
- `requirements.txt` – Python dependencies

---

## 🧠 Example Use Case

> You're a supply manager and want to estimate how many units of Product **A123** will be needed in Store **S001** in **March**.  
> Just enter the month, product, and store — and this model will give you the forecast!

---

## 🙋‍♂️ Want to Improve This?

- Add CSV upload for retraining
- Show historical demand charts
- Add more months/products

Feel free to fork this Space or [contact the author](mailto:ah770643@gmail.com).

---
🌐 Built with 💡 for smart inventory planning.
---
title: "📦 Supply Chain Demand Forecasting"
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.12.0"
app_file: app.py
pinned: false
---

# 📦 Supply Chain Demand Forecaster

This Gradio app predicts product demand based on:

- 📅 **Month**
- 📦 **Product**
- 🏬 **Store**

Built with an XGBoost machine learning model trained on inventory data.

## 🚀 How to Use

1. Select the Month (1–12)
2. Choose Product ID
3. Choose Store ID
4. Click Submit
5. See Predicted Demand 📊

## 🔗 Try It Live

👉 [Open App in Browser](https://huggingface.co/spaces/Ahmedhassan54/Demand-Forecasting)

## 🛠️ Built With

- `Gradio`
- `XGBoost`
- `scikit-learn`
- `pandas`
