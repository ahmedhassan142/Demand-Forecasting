import gradio as gr
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
product_encoder = joblib.load("product_encoder.pkl")
store_encoder = joblib.load("store_encoder.pkl")

product_map = dict(enumerate(product_encoder))
store_map = dict(enumerate(store_encoder))

def predict(month, product_name, store_name):
    product_id = {v: k for k, v in product_map.items()}.get(product_name)
    store_id = {v: k for k, v in store_map.items()}.get(store_name)
    
    if product_id is None or store_id is None:
        return "Invalid input"

    df = pd.DataFrame([[month, product_id, store_id]], columns=["month", "product_id", "store_id"])
    pred = model.predict(df)[0]
    return f"Predicted demand: {round(float(pred), 2)}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Dropdown(list(product_map.values()), label="Product"),
        gr.Dropdown(list(store_map.values()), label="Store"),
    ],
    outputs="text",
    title="Supply Chain Demand Forecaster",
    description="Predict demand based on month, product, and store."
)

if __name__ == "__main__":
    demo.launch()
