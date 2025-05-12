#%%
from groq import Groq
import base64
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

#%% Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#%% Path to your image
image_path = "waldo.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

#%%
client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe where to find Waldo. indicate it as x, y with (0, 0) in the top left corner, and (1, 1) at the bottom right."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

print(chat_completion.choices[0].message.content)
# %% load waldo.jpg
import matplotlib.pyplot as plt
from PIL import Image

waldo = Image.open("waldo.jpg")

# %% indicate a point on (0.47, 0.57) on the image
plt.imshow(waldo)
plt.scatter(0.57 * waldo.size[0], 0.47 * waldo.size[1], color="red", s=100)
plt.show()

# %% what is image resolution?
print(waldo.size)

# %%
