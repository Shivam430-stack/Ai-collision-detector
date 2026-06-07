import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction, Trainer, TrainingArguments
#note:this code shoud be used in google colab with t4 gpu runtime for faster and easier traning 
# google colab workspac path for the laocl storage
WORKSPACE_PATH = "/content/Ai_collision_detector"
os.makedirs(WORKSPACE_PATH, exist_ok=True)

CHECKPOINTS_PATH = os.path.join(WORKSPACE_PATH, "collision_transformer_hf_checkpoints")
MODEL_PATH = os.path.join(WORKSPACE_PATH, "final_collision_model")

# Hyperparameters
MAX_STEPS = 15
TOTAL_FEATURES = 3

class WeightedCollisionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs.get("labels")
        outputs = model(**inputs)

        #the custom generator needs to be applied to the decoders last hidden state
        #outputs.decoder_hidden_states[-1] has shape batch_size, prediction_length,and the  d_model
        #Since prediction_length is 1  itss batch_size, 1, d_model
        #Correctly access the last decoder hidden states
        last_hidden_state = outputs.decoder_hidden_states[-1]
        predictions = model.generator(last_hidden_state) # Apply the custom generator

        current_device = predictions.device
        weights = torch.tensor([1.0, 8.0, 10.0], dtype=torch.float32).to(current_device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(predictions.view(-1, 3), targets.view(-1))

        return (loss, outputs) if return_outputs else loss

class HFCollisionDataset(Dataset):
    def __init__(self, raw_data, seq_len=15):
        self.dataset = raw_data.filter(lambda item: len(item["objects"]["bbox"]) > 0)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # extract the nested Hugging Face 'objects' dictionary
        objects = sample["objects"]
        boxes = objects["bbox"]        #list of boxes [[x, y, w, h], ....]
        classes = objects["category"] #list of labels [2, 0, ...........]

        if len(classes) > 0:
            label_id = classes[0] % 3  # Grab the first detected object's type
            label = torch.tensor(label_id, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
#procces the boxes aroudn the object
        coordinates = []
        valid_steps = []

        for i in range(self.seq_len):
            if i < len(boxes):
                x_pos, y_pos, width, height = boxes[i]
                box_area = width * height
                coordinates.append([float(x_pos), float(y_pos), float(box_area)])
                valid_steps.append(1.0) # 1.0 means this for is a real box aorund a object or a thing
            else:
                if len(boxes) > 0:
                    x_pos, y_pos, width, height = boxes[-1]
                    coordinates.append([float(x_pos), float(y_pos), float(width * height)])
                else:
                    coordinates.append([0.0, 0.0, 0.0])
                valid_steps.append(0.0) # 0.0 means this is for an empty filler padding

        history = torch.tensor(coordinates, dtype=torch.float32)
        mask = torch.tensor(valid_steps, dtype=torch.float32).unsqueeze(-1)

        # Creates a tracking shape [seq_len, 1] containing sequential counting tag
        time_steps = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(-1)

        future_coords = torch.zeros((1, TOTAL_FEATURES), dtype=torch.float32)
        future_times = torch.zeros((1, 1), dtype=torch.float32)

        #reurn the 4 values becuase for the forwards pass expects them
        return {
            "past_values": history,
            "labels": label,
            "past_time_features": time_steps,
            "past_observed_mask": mask,
            "future_values": future_coords,
            "future_time_features": future_times
        }
#laod the datset 
def train_hf_model():
    print("🌐 Downloading trajectory dataset from Hugging Face Hub...")
    downloaded_data = load_dataset("detection-datasets/coco", split="train")

    train_data = HFCollisionDataset(downloaded_data, seq_len=MAX_STEPS)
#the config is set as per the data of the dataset 
    config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=MAX_STEPS,
            input_size=TOTAL_FEATURES,
            d_model=64,
            encoder_layers=2,
            num_labels=3,
            freq=None,
            num_time_features=1,
            lags_sequence=[0], 
            output_hidden_states=True
    )
    model = TimeSeriesTransformerForPrediction(config)
    model.generator = nn.Linear(config.d_model, 3)

    run_settings = TrainingArguments(
        output_dir=CHECKPOINTS_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2
     )

    trainer = WeightedCollisionTrainer(
        model=model,
        args=run_settings,
        train_dataset=train_data
     )

    print(f"🔥 Training has started ,the  temporary directory: {CHECKPOINTS_PATH}")
    trainer.train()

    model.save_pretrained(MODEL_PATH)
    print(f"💾 Complete success! Final weights safely saved to Colab workspace at: {MODEL_PATH}")

if __name__ == "__main__":
    train_hf_model()

