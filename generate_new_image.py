#### Import all necessary libraries ##########
import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL
import time
import random
import os
import pandas as pd

from sklearn import metrics
import torch.nn as nn


import cv2
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from diffusers import DiffusionPipeline
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import re
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import sys


from utils import get_input_image_data, enhance_class_name, segment


DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


folder_name="/home/fa077317/rahat/IMAGENET12/IMAGENET12/Imagenet_10_new/train/"

class_name = ('chickadee',
 'water ouzel',
 'loggerhead',
 'box turtle',
 'garter snake',
 'sea snake',
 'black and gold garden spider',
 'tick',
 'ptarmigan',
 'prairie chicken')

t10_org, all_class = get_input_image_data(folder_name, class_name)

#model_name_or_path = "TheBloke/Synthia-70B-v1.1-GPTQ"
model_name_or_path = "TheBloke/LoKuS-13B-GPTQ"
model_basename = "model"
use_triton = False

##################################################################################
# LLaMA 2
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model_L = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device=DEVICE,
        use_triton=use_triton,
        quantize_config=None)

sys.path.append('/home/fa077317/Grounded_Sam_26_jan/Universal_for_git/GroundingDINO/')  # Replace '/path/to/GroundingDINO/' with the actual path to the directory containing the module

from groundingdino.util.inference import Model

###################################################################################################################
############## Need to download and locate groundingdino_swint_ogc.pth and sam_vit_h_4b8939.pth weights ###########
GROUNDING_DINO_CONFIG_PATH = "/home/fa077317/Grounded_Sam_26_jan/Universal_for_git/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
SAM_CHECKPOINT_PATH = "/home/fa077317/Grounded_Sam_26_jan/weights/sam_vit_h_4b8939.pth"

GROUNDING_DINO_CHECKPOINT_PATH = "/home/fa077317/Grounded_Sam_26_jan/weights/groundingdino_swint_ogc.pth"
###################################################################################################################

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

SAM_ENCODER_VERSION = "vit_h"

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to(DEVICE)


counter=0
failed_list=[]


#############  This should be more generalize version of the class_name ###########################################

prompt_class=('bird',
 'bird',
 'turtle',
 'turtle',
 'snake',
 'snake',
 'spider',
 'tick',
 'bird',
 'chicken')
###################################################################################################################

########### This are the predefined background types for the objects ##############################################
########### You can modify as you want the backgrounds according to the dataset and subjects type #################
back_g_types = [
    ' in a snowy weather',
    ' in a desert',
    ' in a rainy season',
    ' in winter season',
    ' in summer season',
    'under a clear sky',
    'in a tropical climate',
    'amidst the autumn foliage',
    'on a foggy day',
    'at the beach',
    'in a mountainous region',
    'on a sunny day',
    'in a coastal area',
    'under a starry night',
    'in a rural landscape',
    'in the heart of a forest',
    'during a thunderstorm'
]
###################################################################################################################


######### This is an optimization for background prompts generation, if not use avoid word then in automatic  #####
### prompt generation prompt_class can appear and then stable diffusion can generate that as a background #########
######## which is a corruption ####################################################################################

# Convert tuple to string with commas
avoid_word = ', '.join(prompt_class)



for lklk in range(0,len(t10_org)):
    image=t10_org[lklk][0]

    truth_value=t10_org[lklk][1]
    class_name = all_class[truth_value]
    CLASSES = [prompt_class[truth_value]]

    # Convert PIL image to NumPy array
    numpy_array = np.array(image)
    del image
    
    # Convert RGB to BGR
    image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    if len(detections.xyxy)==0:
        failed_list.append(lklk)
        del detections
    else:
        # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _ 
        #     in detections]
        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
    
    
        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1]
        del image
        
        
        masssk = detections.mask[0].astype('int8')
        del detections
        masssk_reshaped = masssk[:, :, np.newaxis]
        
        # Ensure masssk_reshaped has the same number of channels as image_rgb
        masssk_reshaped = np.repeat(masssk_reshaped, 3, axis=-1)
        
        
        # Element-wise multiplication
        image_up = image_rgb * masssk_reshaped

        # image_up = np.fliplr(image_up)    ####################### Flip right ##################
        
        del image_rgb
        del masssk_reshaped
    
    
        prompt_select=["What would be two descriptive prompts describing the perfect background", "Give me two short and vivid prompts that depict the ideal background", "In short Please give me two expressive prompts that capture the perfect background"]

        prompt = random.choice(prompt_select) + random.choice(back_g_types) + ' within one line without using '+ 'any of the words such as '+ avoid_word  + 'in a consise manner'
        
                
        prompt_template=f'''Below is an instruction that describes a task. Write a response that appropriately completes the request in short.
        
        ### Instruction: {prompt}
        
        ### Response:
        '''
        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(DEVICE) ## Something wrong
        output = model_L.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
        del input_ids
        out_str_o = tokenizer.decode(output[0])
        #print(out_str_o)
        del output
        
        
        
        prompts = []
        resp = out_str_o.split('###')[-1]
        del out_str_o
        a_resp = resp.split('\n')
        
        for i in range(1,len(a_resp)):
            # print(a_resp[i])
            '''if a_resp[i] == '':
                continue'''
            if re.search("^[0-9].",a_resp[i].strip()):
                # print('found')
                re_v = re.search("^[0-9].",a_resp[i].strip())
                prompts.append(a_resp[i].split(re_v.group())[1].replace('</s>',''))
        
        prompt = random.choice(prompts)
        print(prompt)
        del prompts
        
        image = pipe(prompt=prompt).images[0]
    
        image_rgb_background = np.array(image)
        del image
        # 'cuda:2'
        # Resize image_rgb to the same size as org_image
        image_rgb_resized = cv2.resize(image_rgb_background, (image_up.shape[1], image_up.shape[0]))
        del image_rgb_background
        
        
        # Create a condition based on black pixels in image_up
        condition = np.all(image_up == [0, 0, 0], axis=-1)
        
        # Combine org_image and image_up based on the condition
        result = np.where(condition[:, :, np.newaxis], image_rgb_resized, image_up)
        del condition
        
        
        
        # plt.axis('off')
        print(lklk)
        ##################################### Change the target directory according to the user #########################
        target_directory = './augment_test_test/' + t10_org.wnids[t10_org.class_to_idx[class_name]]
        #################################################################################################################
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        # Plot and save the figure
        plt.imshow(result)
        del result
        plt.axis('off')
        plt.savefig(os.path.join(target_directory, f'{t10_org.wnids[t10_org.class_to_idx[class_name]]}_aug_{lklk}_{counter}.png'),bbox_inches='tight', pad_inches=0)
        counter=counter+1
        # plt.show()
