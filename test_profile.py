import time
import torch
from musetalk.utils.utils import load_all_model

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device('cuda:0')
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device
)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
timesteps = torch.tensor([0], device=device)

def test_batch(bs):
    latent_batch = torch.randn(bs, 8, 32, 32, dtype=torch.float16, device=device)
    audio_feature_batch = torch.randn(bs, 50, 384, dtype=torch.float16, device=device)
    for _ in range(3):
        pred = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        pred = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Batch={bs}: Time={end - start:.3f}s FPS={10 * bs / (end - start):.1f}")

test_batch(2)
test_batch(4)
test_batch(8)
test_batch(16)
