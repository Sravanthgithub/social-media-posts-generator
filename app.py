import torch
from diffusers import StableDiffusionPipeline
import openai
import re 
import gradio as gr
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from copy import deepcopy
from utils import memeify_image, cleanup_caption
from utils import NEG_PROMPT, OPENAI_TOKEN
import os
from io import BytesIO
CWD = os.getcwd()


MEME_FONT_PATH = os.path.join(CWD,  'fonts', 'impact.ttf')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "prompthero/openjourney"
PIPE = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)

PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


openai.api_key = OPENAI_TOKEN


def post_generator(task : str, image_desc :str, theme :str, text_pos :str):
    
    image = PIPE(
                  image_desc,
                  negative_prompt=NEG_PROMPT,
                ).images[0]

    content_mapper = {
        'Memes': f"""
                Generate a super funny caption for this:
                Image description: {image_desc}
                Theme: {theme}
                   """,
        'Inspirational Quotes': f"""
                Generate an inspirational quote for this:
                Image description: {image_desc}
                Theme: {theme}
                   """,
        'Slogans': f"""
                Generate an impactfull slogan for this:
                Image description: {image_desc}
                Theme: {theme}
                   """,
        'Jokes': f"""
                Generate a joke for this:
                Image description: {image_desc}
                Theme: {theme}
                   """
    }

    content = content_mapper[task]
    
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "assistant", "content": content}
        ]
    )

    text = response.choices[0]['message']['content']

    text = cleanup_caption(text)

    if text_pos == 'top':
        top = text
        bottom = ''
    else:
        top = ''
        bottom = text

    final_img = memeify_image(image, top=top, bottom=bottom)

    return final_img


def text_generator(task : str, image, theme:str):
    """
    input - image, theme and return text
    """

    inputs = PROCESSOR(image, return_tensors="pt")

    out = MODEL.generate(**inputs)
    image_desc = PROCESSOR.decode(out[0], skip_special_tokens=True)

    content_mapper = {
        "Story": f"""
                Generate a story for this:
                Image description: {image_desc}
                Theme: {theme}
                     """,
        "Poem": f"""
                Generate a poem for this:
                Image description: {image_desc}
                Theme: {theme}
                        """,
        "Lyrics": f"""
                Generate lyrics for this:
                Image description: {image_desc}
                Theme: {theme}
                        """
    }

    content = content_mapper[task]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "assistant", "content": content}
        ]
    )

    text = response.choices[0]['message']['content']

    return text




    


                

