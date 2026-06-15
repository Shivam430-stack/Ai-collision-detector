import os
import cv2
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from safetensors.torch import load_file

FRAMES_TRACKED = 15
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_weights")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model.safetensors")

config = TimeSeriesTransformerConfig.from_json_file(CONFIG_PATH)
config.output_hidden_states = True

model = TimeSeriesTransformerForPrediction(config)
model.generator = nn.Linear(model.config.d_model, 3)

state_dict = load_file(WEIGHTS_PATH)
model.load_state_dict(state_dict, strict=False)
model.eval()

CHANNELS = config.num_time_features
history = [[0.0, 0.0, 0.0] for _ in range(FRAMES_TRACKED)]

def process_webcam_frame(frame):
    global history
    if frame is None:
        return None
        
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cx, cy, area = 0.0, 0.0, 0.0
    if contours:
        big_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big_contour) > 400:
            x, y, w, h = cv2.boundingRect(big_contour)
            cx, cy, area = float(x), float(y), float(w * h)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    history.append([cx, cy, area])
    history.pop(0)
    
    hist_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.ones((1, FRAMES_TRACKED, 3), dtype=torch.float32)
    
    time_steps = torch.arange(FRAMES_TRACKED, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    time_tensor = time_steps.repeat(1, 1, CHANNELS)
    
    with torch.no_grad():
        outputs = model(
            past_values=hist_tensor,
            past_time_features=time_tensor,
            past_observed_mask=mask_tensor,
            future_values=torch.zeros((1, 1, 3), dtype=torch.float32),
            future_time_features=torch.zeros((1, 1, CHANNELS), dtype=torch.float32)
        )
        preds = model.generator(outputs.decoder_hidden_states[-1])
        logits = preds.view(-1, 3).squeeze(0)
        
        safe_score = logits[0].item()
        caution_score = logits[1].item()
        collision_score = logits[2].item()
        
        desensitizer_bias = 2.0
        
        if collision_score > (safe_score + desensitizer_bias) and collision_score > caution_score:
            pred_class = 2
        elif caution_score > safe_score:
            pred_class = 1
        else:
            pred_class = 0
            
    statuses = ["SAFE", "WARNING", "COLLISION WARN!"]
    ui_colors = [(0, 255, 0), (0, 150, 255), (0, 0, 255)]
    
    cv2.putText(frame_bgr, f"AI status: {statuses[pred_class]}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_colors[pred_class], 3)
    
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

with gr.Blocks(title="AI Collision Stream Portal") as demo:
    gr.Markdown("## 🚗 Real-Time Transformer Video Stream Collision Portal")
    
    with gr.Row():
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Input")
        video_output = gr.Image(label="AI Model Analysis Stream")
        
    webcam_input.stream(
        fn=process_webcam_frame, 
        inputs=webcam_input, 
        outputs=video_output,
        time_limit=30, 
        stream_every=0.1
    )

if __name__ == "__main__":
    demo.launch()
