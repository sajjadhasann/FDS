import gradio as gr
import pandas as pd
import numpy as np
import joblib


df = pd.read_csv("./data/test_dataset.csv")
gbc = joblib.load('./models/gbc_model.joblib')
xgb = joblib.load('./models/xgb_model.joblib')
rf = joblib.load('./models/rf_model.joblib')
ann = joblib.load('models/ann_model.pkl')

def predict(index):
    unlabeled_data = df.drop(columns=['Class'])
    transaction = unlabeled_data.iloc[index]
    
    gbc_prediction = gbc.predict([transaction.values])
    xgb_prediction = xgb.predict([transaction.values])
    rf_prediction = rf.predict([transaction.values])
    ann_prediction = ann.predict([transaction.values])

    gbc_result = "Fraudulent" if gbc_prediction[0] else "Legitimate"
    xgb_result = "Fraudulent" if xgb_prediction[0] else "Legitimate"
    rf_result = "Fraudulent" if rf_prediction[0] else "Legitimate"
    ann_result = "Fraudulent" if ann_prediction[0] else "Legitimate"
    return f"GBC: {gbc_result} | XGB: {xgb_result} | RF: {rf_result} | ANN: {ann_result}"  


css = """
.gradio-container-4-28-3 .prose h1{color:#00e6ff !important; font-size:140px !important},
.app.svelte-182fdeq.svelte-182fdeq {
    padding: 10%
},
.datatable{height:50% !important},
.prediction-button, .secondary.svelte-cmf5ev{background: linear-gradient(90deg, #48c6ef 0%, #6f86d6 100%) !important; color: white;},
"""

with gr.Blocks(css=css) as demo:
    markdown = gr.Markdown(value="# Fraud Detection System <p style='font-size:24px'>Select a row from the DataFrame to predict if it's fraudulent or legitimate. <br/><br/> <a href='https://sajjad.design' style='color:#ffc107'>Sajjad Hasan's</a> graduation project, supervised by Dr. Hasanein Yaarub.<br/><a href='mailto:contact@sajjad.design' style='color:#ffc107'>contact@sajjad.design</a><br/><a href='https://sajjad.design/fds' style='color:#ffc107'>Read about project</a><p/></div>")

    data = gr.Dataframe(value=df, headers=list(df.columns), interactive=False, elem_classes=["datatable"])
    index = gr.State()
    button = gr.Button("Predict",elem_classes=["prediction-button"])
    result = gr.Label(label="Result")
    
    layout = [
        [markdown],
        [data],
        [button],
        [result]
    ]
    
    button.click(fn=predict, inputs=index, outputs=result)

    def select_trans(evt: gr.SelectData):
        return evt.index[0]
    
    data.select(select_trans, None, index)

demo.launch()
