# social-media-posts-generator

## Description

MERCOR -AI TRACK (2)

This is a simple social media posts generator. It can create posts for any time of Social Media. Ex - Memes, Inspirational Quotes, Slogans, Jokes etc. 

This can also create Stories, Poems, Songs based on an image :D (Yes, you heard it right!)

#### Deployed API End Point - https://huggingface.co/spaces/Sravanth/Social-Media-Post-Generator

But since our project uses huge models and we deployed on CPU instances, the UI gets disconned every time in hugging face. So we recommend using it as an API Endpoint in your own files/projects.

### Requirements to run as an API Endpoint 

```
pip install gradio_client pillow
```

```python
from PIL import Image
from gradio_client import Client
client = Client("https://sravanth-social-media-post-generator.hf.space/")
result = client.predict(
            "Memes",	# str (Option from: ['Memes', 'Jokes', 'Slogans', 'Inspirational Quotes']) in 'Category' Dropdown component
            "A kid crying just after returning from school",	# str  in 'Image Description/prompt' Textbox component
            "Bad results",	# str  in 'Theme' Textbox component
            "top",	# str (Option from: ['top', 'bottom']) in 'text_position' Dropdown component
            fn_index=0
)

image_path = result[0]
image = Image.open(image_path)
image.show()
```
### Here is the image generated
![image](https://github.com/Sravanthgithub/social-media-posts-generator/assets/77894804/fc9caf9f-0c45-45a8-bc53-5dd86c23303e)


```python
import requests
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

client = Client("https://sravanth-social-media-post-generator.hf.space/")
result = client.predict(
            "Poem",	# str (Option from: ['Story', 'Lyrics', 'Poem']) in 'Category' Dropdown component
            img_url,	# str (filepath or URL to image) in 'parameter_17' Image component
            "Beautiful Life",	# str  in 'Theme' Textbox component
            fn_index=1
)
print(result)
``` 
### Here is the poem generated

```
A woman sits upon the shore,
Her dog beside her, evermore,
The waves crash gently at her feet,
As the sun begins to take its seat.

With phone in hand, she smiles bright,
Capturing this gorgeous sight,
A moment frozen, forever caught,
A beautiful life that can't be bought.

The ocean's melody fills her ears,
Washing away all her fears,
A symphony of tranquility,
In this moment, life's simplicity.

Her furry friend, by her side,
Fills her heart with love and pride,
A loyal companion, always true,
A bond unbreakable between the two.

The wind dances through her hair,
Carrying away any trace of despair,
As she reflects on life's sweet pace,
And all the joy she's learned to embrace.

In this picture-perfect scene,
She's reminded of what life can mean,
It's not the things that we possess,
But the moments that bring us happiness.

So on this beach, with dog and phone,
She knows she's never really alone,
For life's beauty surrounds her here,
And fills her heart with love and cheer.
```


