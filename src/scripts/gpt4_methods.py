import os
import torch
import clip
import requests
import json
import numpy as np
import pandas as pd
import sentence_transformers
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from PIL import Image
from openai import OpenAI


class clip_based_method():
    def __init__(self, args):
        self.device = args.device
        self.clip_model_type = args.clip_model_type
        self.clip_model = self.get_clip_model(self.clip_model_type)
        self.feature_size = self.get_feature_size(self.clip_model_type)
        
    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model
    
    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError

    def cal_i2t_cosine_similarity(self, image, text):
        image_feature = self.clip_model.encode_image(image).to(self.device)
        text_feature = self.clip_model.encode_text(text).to(self.device)
        
        i2t_cosine_similartiy = torch.nn.functional.cosine_similarity(image_feature, text_feature).item()
        return i2t_cosine_similartiy


class sentence_transformer_based_method():
    def __init__(self, args):
        self.device = args.device
        self.model_name = args.sentence_transformer_model_name
        self.sentence_model = self.get_sentence_transformer_model(self.model_name)
        
    def get_sentence_transformer_model(self, model_name):
        model = sentence_transformers.SentenceTransformer(model_name)
        return model
    
    def cal_t2t_cosine_similarity(self, image_label, label):
        # image_label 'GPT4 generates the image label'
        image_feature = self.sentence_model.encode(image_label)
        text_feature = self.sentence_model.encode(label)
        
        t2t_cosine_similarity = sentence_transformers.util.cos_sim(image_feature, text_feature)
        return t2t_cosine_similarity.tolist()[0][0]

    
class gpt4_based_method():
    def __init__(self, args):
        self.client = OpenAI(api_key=args.gpt4_api_key)
        self.single_image_prompt = args.single_image_prompt
        self.image_mask_prompt = args.image_mask_prompt
        self.scene_image_prompt = args.scene_image_prompt
        self.str_template = args.str_template
        self.str_ins2cormap = args.str_ins2cormap
        self.smms_server = SMMS(args)
        
    def get_single_image_label(self, image):
        image_url = self.smms_server.upload(image)
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.single_image_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        }
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0]
    
    def get_image_mask_label(self, image, mask):
        image_url = self.smms_server.upload(image, 'temp_image')
        mask_url = self.smms_server.upload(mask, 'temp_mask')
        # print(image_url)
        # print(mask_url)
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.image_mask_prompt + 'The template_json format is like ' + self.str_template,
                        },
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                        {
                            "type": "image_url",
                            "image_url": mask_url,
                        }
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0]

    def get_scene_objects_label(self, image, mask):
        image_url = self.smms_server.upload(image, 'temp_image')
        mask_url = self.smms_server.upload(mask, 'temp_mask')
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.scene_image_prompt + 'The template_json format is like' + self.str_template + '. And the instance_color_map is ' + self.str_ins2cormap,
                        },
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                        {
                            "type": "image_url",
                            "image_url": mask_url,
                        }
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0]
        
    
    
class SMMS(object):
    def __init__(self, args):
        # nvlcvIWuj2EksGWguPvrzTQLV5ULzrYf
        self.headers = {'Authorization': args.smms_api_key}
        
    def upload(self, image, temp_name='temp_file'):
        image.save(temp_name + '.jpg')
        image = {'smfile': open(temp_name + '.jpg', 'rb')}
        url = 'https://sm.ms/api/v2/upload'
        res = requests.post(url, files=image, headers=self.headers, timeout=5).json()
        os.remove(temp_name + '.jpg')
        if res['success'] == False:
            return res['images']
        return res['data']['url']
    
    def getHistory(self):
        url = 'https://sm.ms/api/v2/upload_history'
        res = requests.get(url, headers=self.headers, timeout=5).json()
        return res

    def deleteHistory(self):
        history = self.getHistory()
        lastTime = time.time()
        for item in history["data"]:
            if item["created_at"] < lastTime:
                url = 'https://sm.ms/api/v2/delete/{}'.format(item["hash"])
                res = requests.get(url, headers=self.headers, timeout=5).json()
        print("History deleted!")
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    # online picture upload key
    parser.add_argument("--smms_api_key", type=str, default="your-own-smms_key")
    # gpt4 params
    parser.add_argument("--single_image_prompt", type=str, default="")
    # 场景照片（第一张）中对应的mask图像（第二张）所表示的物体的名称、颜色和描述是什么？他常见的重量、碰撞体是什么？请根据template_json的格式回答
    parser.add_argument('--image_mask_prompt', type=str, default="What are the name, color, and description of the object represented in the scene photo (first image) and its corresponding mask image (second image)? What is its common weight and collider? Please answer according to the template_json format.")
    # 场景照片（第一张）中对应的mask图像（第二张）所表示的物体们（mask和instance_id的对应参照instance_color_map）的名称、颜色和描述是什么？他常见的重量、碰撞体是什么？请为每一个instance生成一个根据template_json的格式回答。
    parser.add_argument('--scene_image_prompt', type=str, default="What are the names, colors, and descriptions of the objects represented by the mask image (second image) corresponding to the scene photo (first image)? Please refer to the instance_color_map to associate the masks with their respective instance_ids. What are their common weights and colliders? Please generate a response for each instance according to the template_json format.")
    parser.add_argument("--gpt4_api_key", type=str, default="your-own-gpt4-key")
    # sentence transformer params
    parser.add_argument("--sentence_transformer_model_name", type=str, default="paraphrase-MiniLM-L6-v2")
    # clip params
    parser.add_argument("--clip_model_type", type=str, default="ViT-L/14@336px")
    # other params
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    

    with open('instance_color_map.json', "r") as file:
        instance_color_map = json.load(file)
    with open('template.json', "r") as file:
        template_json = json.load(file)

    args.str_template = str(template_json)
    args.str_ins2cormap = str(instance_color_map)

    smms_model = SMMS(args)
    gpt_model = gpt4_based_method(args)
    
    print(gpt_model.get_image_mask_label(Image.open('frame_00001.png'), Image.open('hat_00001.png')))
    print(gpt_model.get_scene_objects_label(Image.open('frame_00001.png'), Image.open('mask_00001.png')))