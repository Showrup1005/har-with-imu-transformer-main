import torch
import json
from models.IMUTransformerEncoder import IMUTransformerEncoder

print("Loading best federated model...")

with open('config.json', 'r') as f:
    config = json.load(f)


model_path = "best_global_model_r5.pth"

model = IMUTransformerEncoder(config)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# ================== WRAPPER FOR RPI ==================
class RPiHARModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.input_proj = original_model.input_proj
        self.transformer_encoder = original_model.transformer_encoder
        self.cls_token = original_model.cls_token
        self.position_embed = getattr(original_model, 'position_embed', None)
        self.encode_position = original_model.encode_position
        self.imu_head = original_model.imu_head
        self.log_softmax = original_model.log_softmax

    def forward(self, x: torch.Tensor):
        src = self.input_proj(x.transpose(1, 2)).permute(2, 0, 1)

        batch_size = src.shape[1]
        cls_token = self.cls_token.unsqueeze(1).repeat(1, batch_size, 1)
        src = torch.cat([cls_token, src], dim=0)

        if self.encode_position and self.position_embed is not None:
            src = src + self.position_embed

        output = self.transformer_encoder(src)
        cls_output = output[0]

        logits = self.imu_head(cls_output)
        return self.log_softmax(logits)

# Export
print("Exporting model...")
mobile_model = RPiHARModel(model)
scripted_model = torch.jit.script(mobile_model)
scripted_model.save("har_imu_rpi_model.pt")

print("Model exported successfully for Raspberry Pi!")
print("File created: har_imu_rpi_model.pt")
