from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt):
    print("⏳ Loading model... (first time will take time)")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    
    pipe = pipe.to("cpu")

    print("🎨 Generating image...")

    image = pipe(prompt).images[0]

    image.save("generated_image.png")
    print("✅ Image saved as generated_image.png")

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    generate_image(user_prompt)