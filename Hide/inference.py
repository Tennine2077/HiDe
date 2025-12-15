import os
from transformers import AutoTokenizer, AutoProcessor
from modeling_qwen2_5_vl_re_infer import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
from is_attention_focused import *
import json
import torch.multiprocessing as mp
import multiprocessing
from joblib import Parallel, delayed
import time
import random
from PIL import Image
import io
import numpy as np
import base64
import gc
import base64
import multiprocessing
from multiprocessing import Pool
from once_inference import messages2out,messages2att,from_img_and_att_get_cropbox,get_inputs
import shutil
import cv2

def once_infer(model,att_processor,ans_processor,sample,messages,img_url,ori_img_url,ques,sig,thre):
    #得到att
    prompt_output_text = [""]
    ques = messages[-1]["content"][-1]["text"]
    prompt_ques = '''
You are a highly precise language analysis engine. Your sole function is to extract entities (e.g., objects, people) from a user's question, and deconstruct them into a canonical, attribute-based format by strictly following a set of rules and a thinking process.

### Thinking Process
Before generating the final output, you must internally follow these steps in order:

1.  **Identify Core Entities**: Read the entire question and identify all key noun phrases. For example, "the green surfboard," "the purple umbrella."
2.  **Deconstruct Attributes for Each Entity Individually**: **This is the critical step. Before considering the relationship between entities, look at each entity in isolation and apply Rules 2, 3, and 4 to fully deconstruct its attributes.**
    *   For instance, first process "the green surfboard" using Rule 2 to get `surfboard with green color`.
    *   Then, process "the purple umbrella" using Rule 2 to get `umbrella with purple color`.
3.  **Handle Relationships Between Entities**: After all entities have been individually deconstructed, check for spatial or logical relationships between them (Rule 5). If a relationship exists, you will list the **already deconstructed** entities as separate items.
4.  **Assemble and Normalize**:
    *   Gather all the canonical entity strings you have transformed.
    *   Convert all text to lowercase.
    *   Join all entities into a single line, separated by a comma and a space (", ").
5.  **Final Formatting**: Enclose the resulting single-line string within the `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.

### Extraction Rules

**Rule 1: Simple Entities**
If a noun is not described by any modifiers, extract the noun itself.
*   *Example*: "the scooter" becomes `scooter`.

**Rule 2: Adjective Attribute Deconstruction**
If an entity is modified by one or more adjectives, the format must be `noun with [property] [type]`. Chain multiple properties consecutively.
*   **Format**: `noun with [property1] [type1] with [property2] [type2]`
*   **Common Type Mappings**:
    *   Colors (red, blue, black, silver) -> `color`
    *   Sizes (large, small, big) -> `size`
    *   Materials (wooden, metal, plastic) -> `material`
*   *Example*: "the large blue truck" becomes `truck with large size with blue color`.

**Rule 3: Possessive Inversion**
Convert all possessive forms (e.g., `X's Y`) uniformly into the `Y of X` format.
*   *Example*: "the woman's handbag" becomes `handbag of woman`.

**Rule 4: Attributive Prepositional Phrases**
If a prepositional phrase describes a component of an entity (e.g., "in a shirt"), preserve the structure and recursively apply the rules to the entity within the phrase.
*   *Example*: "the man in the green shirt" becomes `man in a shirt with green color`.

**Rule 5: Relational Prepositional Phrases**
If a prepositional phrase describes a relationship between two separate entities (e.g., `next to`, `on the left of`), **extract the entities as separate items, but only after each entity has been fully deconstructed according to the other rules.** Do not include the relational words in the output.
*   *Example 1 (Simple)*: "the dog on the left side of the scooter" becomes `dog, scooter`.
*   **Example 2 (With Attributes)**: **"Is the green surfboard on the left side of the purple umbrella?" becomes `surfboard with green color, umbrella with purple color`.** (Note: The surfboard and umbrella are first deconstructed individually, then listed as separate entities).

**Rule 6: Compound Nouns**
Recognized compound nouns should be treated as a single entity.
*   *Example*: "the traffic light" becomes `traffic light`.

### Output Format Rules
1.  **Sole Output**: The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.
2.  **Single-Line Format**: The output must be a single continuous line of text.
3.  **Delimiter**: Multiple entities must be separated by a comma followed by a space (", ").
4.  **Lowercase**: All output characters must be in lowercase.
5.  **Content Exclusion**: The final entity string must not include articles (a, an, the), question words (what, which, is), or purely relational words (side, next to, of).

Now, following all the rules above, extract the entities from the question below:
{input_text}
'''

    prompt_messages = [{"role": "user","content": [{"type": "text", "text": prompt_ques.format(input_text=ques.replace("\nAnswer with the option's letter from the given choices directly.","").replace("\nAnswer the question using a single word or phrase.","").replace("\nAnswer with YES or NO directly:",""))}],},]
    prompt_output_text,_ = messages2out(prompt_messages,model,ans_processor)
    answer_out = prompt_output_text[0].split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
    messages[-1]["content"] = messages[-1]["content"][:-1]
    # messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + ques})
    outputs = {}
    
    #如果提取出了实体词
    if answer_out: 
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: "+answer_out})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(inputs,model,att_processor)  # Retrieve attention from model outputs
        results = from_img_and_att_get_cropbox(messages,att_processor,attention, idx2word_dicts, img_url, img_start, img_end,sig,thre)
        for s in sig:
            for t in thre:
                img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes = results[str(s)][str(t)]
                messages = [ {"role": "user","content": [],},]
                # # #加上原图
                for img in ori_img_url:
                    messages[-1]["content"].append({"type": "image", "image": img})
                #加上这次处理新出的图
                for h_img in highlight_imgs:
                    messages[-1]["content"].append({"type": "image", "image": h_img})
                #加上问题
                messages[-1]["content"].append({"type": "text", "text": ques})
                output_text,_ = messages2out(messages,model,ans_processor)
                if not str(s) in outputs:outputs[str(s)] = {}
                outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes]
                
    #没有提取出实体词
    else:
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + ques})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(inputs,model,att_processor)  # Retrieve attention from model outputs
        results = from_img_and_att_get_cropbox(messages,att_processor,attention, idx2word_dicts, img_url, img_start, img_end,sig,thre)
        for s in sig:
            for t in thre:
                img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes = results[str(s)][str(t)]
                messages = [ {"role": "user","content": [],},]
                #加上原图
                for img in ori_img_url:
                    messages[-1]["content"].append({"type": "image", "image": img})
                #加上这次处理新出的图
                for h_img in highlight_imgs:
                    messages[-1]["content"].append({"type": "image", "image": h_img})
                #加上问题
                messages[-1]["content"].append({"type": "text", "text": ques})
                output_text,_ = messages2out(messages,model,ans_processor)
                if not str(s) in outputs:outputs[str(s)] = {}
                outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes]
    return outputs

def cycle_epoch_infer(gpu_id,rank,dataset_part,savedir,max_pixels,sig,thre):
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    device = f"cuda:{gpu_id}"

    print(rank,len(dataset_part),device)

    model_path = r"/home/luohaoyu.lhy/Shijiu/Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration_re.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # load_in_8bit=True,
        device_map=device
    )

    qwen_processor = AutoProcessor.from_pretrained(model_path,use_fast=True,min_pixels=256*28*28,max_pixels=max_pixels*28*28)

    for sample in tqdm(dataset_part):
        results = sample
        img_url = [sample["image"]]
        ori_img_url = []
        for img in img_url:
            ori_img_url.append(img)
        messages = [
                {
                    "role": "user",
                    "content": [],
                },
            ]
        for img in img_url:
            messages[-1]["content"].append({"type": "image", "image": img})

        ques = sample["Text"]

        #直接回答
        messages[-1]["content"].append({"type": "text", "text": ques})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,processor,model)
        output_text,end_ques = messages2out(messages,model,qwen_processor)
        results["answer"] = {}
        results["answer"]["ori"] = output_text[0]
        results["bounding_box"] = {}
        results["prompt_text"] = {}
        torch.cuda.empty_cache()
        
        #先进行att计算，再回答
        outputs = once_infer(model,att_processor,ans_processor,sample,messages,img_url,ori_img_url,ques,sig,thre)
        for s in sig:
            for t in thre:
                prompt_output_text,output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes = outputs[str(s)][str(t)]
                results["answer"][f"TAD_{i+1}_s{s}_t{t}"] = output_text[0]
                results["prompt_text"][f"TAD_{i+1}"] = prompt_output_text[0]
                results["bounding_box"][f"TAD_{i+1}_s{s}_t{t}"] = bounding_boxes
        
        #保存答案
        serialize_dict(results,savedir)
        torch.cuda.empty_cache()
        print(savedir)
    del model
    torch.cuda.empty_cache()
