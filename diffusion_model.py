from transformers import CLIPTokenizer, CLIPTextModel
import torch

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

prompt = ["a red sports car under sunset"]
# prompt = ["a b c d e f ac ca"]
inputs = tokenizer(prompt, padding=True, return_tensors="pt")

with torch.no_grad():
    text_embeds = text_encoder(**inputs).last_hidden_state

print(text_embeds.shape)  # (1, 77, 768)








from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a fish jumping toward a boat in ocean, ultra-realistic"
image = pipe(prompt).images[0]
image.save("gen1.png")
