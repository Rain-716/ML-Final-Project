from pathlib import Path
import joblib
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text, safe_response

root = Path(__file__).resolve().parents[1]
artifact = joblib.load(root / 'outputs' / 'best_model.joblib')
model = artifact['model']; le = artifact['label_encoder']; responses = artifact['responses']
vectorizer = model.named_steps['tfidf']
resp_X = vectorizer.transform([r['text_clean'] for r in responses])

def chat(user_text, history=None):
    x = clean_text(user_text)
    pred_id = model.predict([x])[0]
    label = le.inverse_transform([pred_id])[0]
    sims = cosine_similarity(vectorizer.transform([x]), resp_X).ravel()
    idx = sims.argmax()
    retrieved = responses[idx]['response']
    label, reply = safe_response(user_text, label, retrieved)
    return f'【识别类别】{label}\n\n{reply}'

with gr.Blocks(title='AI心理问诊对话 Demo') as demo:
    gr.Markdown('# AI心理问诊对话 Demo\n输入你的困扰，系统会识别心理支持场景并给出安全回应。\n\n⚠️ 本Demo不替代专业心理咨询或医疗诊断。')
    inp = gr.Textbox(label='用户输入', lines=3, placeholder='例如：I feel anxious and cannot sleep before exams')
    out = gr.Textbox(label='系统回应', lines=6)
    gr.Button('发送').click(chat, inp, out)
if __name__ == '__main__': demo.launch()
