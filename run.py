import os
import cv2
import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from safetensors.torch import load_file

# --- NATIVE FILE LOADING USING OS MODULE ---
FRAMES_TRACKED = 15
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_weights")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model.safetensors")

print("🤖 loading model structure from config.json...")
config = TimeSeriesTransformerConfig.from_json_file(CONFIG_PATH)
config.output_hidden_states = True  # make sure hidden states are on

model = TimeSeriesTransformerForPrediction(config)
model.generator = nn.Linear(model.config.d_model, 3)

print("💾 loading raw model.safetensors weights into layers...")
state_dict = load_file(WEIGHTS_PATH)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ model loaded successfully with zero internet checks!")

# --- START WEBCAM LOOP ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️ camera not found! running quick fake data simulation test...")
    fake_history = [[100.0 + (i*2), 150.0, 300.0] for i in range(FRAMES_TRACKED)]
    
    h_tensor = torch.tensor(fake_history, dtype=torch.float32).unsqueeze(0)
    m_tensor = torch.ones((1, FRAMES_TRACKED, 1), dtype=torch.float32)
    t_tensor = torch.arange(FRAMES_TRACKED, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        out = model(past_values=h_tensor, past_time_features=t_tensor, past_observed_mask=m_tensor,
                    future_values=torch.zeros((1, 1, 3)), future_time_features=torch.zeros((1, 1, 1)))
        logits = model.generator(out.decoder_hidden_states[-1])
        print(f"simulated pred: class {torch.argmax(logits.view(-1, 3), dim=-1).item()}")
    exit()

print("🎥 camera feed is live! press 'q' to stop.")

history = [[0.0, 0.0, 0.0] for _ in range(FRAMES_TRACKED)]

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cx, cy, area = 0.0, 0.0, 0.0
    
    if contours:
        # find the biggest moving object (like the pillow or your elbow)
        big_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big_contour) > 400:
            x, y, w, h = cv2.boundingRect(big_contour)
            cx, cy, area = float(x), float(y), float(w * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    # slide the window: add new frame data, drop oldest
    history.append([cx, cy, area])
    history.pop(0)
    
    hist_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.ones((1, FRAMES_TRACKED, 1), dtype=torch.float32)
    time_tensor = torch.arange(FRAMES_TRACKED, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        outputs = model(
            past_values=hist_tensor,
            past_time_features=time_tensor,
            past_observed_mask=mask_tensor,
            future_values=torch.zeros((1, 1, 3), dtype=torch.float32),
            future_time_features=torch.zeros((1, 1, 1), dtype=torch.float32)
        )
        preds = model.generator(outputs.decoder_hidden_states[-1])
        logits = preds.view(-1, 3).squeeze(0)
        
        # separate the raw classification scores
        safe_score = logits[0].item()
        caution_score = logits[1].item()
        collision_score = logits[2].item()
        
        # --- ADJUSTABLE BIAS ZONE ---
        # higher number = less sensitive. change 2.5 to 4.0 if it panics too easily!
        desensitizer_bias = 2.0 #THE BIAS of 2.0 is final and is the most optimal and accurate bias whihc prevents mismeasure and accurate results and verifed though many runs on diffrent biases.
        
        # custom check to prevent over-sensitive triggers
        if collision_score > (safe_score + desensitizer_bias) and collision_score > caution_score:
            pred_class = 2  # COLLISION WARN!
        elif caution_score > safe_score:
            pred_class = 1  # WARNIng
        else:
            pred_class = 0  # SAFE
        
    statuses = ["SAFE", "WARNING", "COLLISION WARN!"]
    ui_colors = [(0, 255, 0), (0, 150, 255), (0, 0, 255)]
    
    cv2.putText(frame, f"AI status: {statuses[pred_class]}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_colors[pred_class], 3)
    
    cv2.imshow("stardance model test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 done!")
