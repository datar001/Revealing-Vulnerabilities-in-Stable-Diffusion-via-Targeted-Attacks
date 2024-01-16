import torch
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import numpy as np
import pdb

class OSModel(nn.Module):
    def __init__(self, model, clip_id_or_path, object_path, device):
        super(OSModel, self).__init__()
        with open(object_path, "r") as f:
            objects = f.readlines()
        self.objects = [obj.strip() for obj in objects]
        self.object_num = len(self.objects)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(clip_id_or_path)
        self.model.to(device)

        self.device = device

    def replace_object(self, ori_prompt, object_id_or_name, ref_num=10):
        if type(object_id_or_name) == int:
            ori_object = self.objects[object_id_or_name]
        else:
            ori_object = object_id_or_name

        assert ref_num < self.object_num
        selected_index = np.random.choice(np.arange(self.object_num), ref_num, replace=False)
        ref_prompts = []
        for id in selected_index:
            ref_prompts.append(ori_prompt.replace(ori_object, self.objects[id]))
        return ref_prompts

    def get_prompt_feature(self, prompts):
        with torch.no_grad():
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
            text_embeds = self.model.get_text_features(inputs.input_ids.to(self.device))
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            # text_embeds = text_embeds.to(self.device)
        return text_embeds

    def forward(self, ori_prompt, object_id_or_name, ref_num=10, thres=None):
        ref_prompts = self.replace_object(ori_prompt, object_id_or_name, ref_num)
        ref_features = self.get_prompt_feature(ref_prompts)
        ori_feature = self.get_prompt_feature(ori_prompt)
        
        diff_features = ref_features - ori_feature.repeat(ref_num, 1)
        diff_sign = torch.sign(diff_features).sum(0)
        if thres is None:
            thres = ref_num -1
        mask = torch.zeros_like(diff_sign).to(self.device)
        mask[abs(diff_sign) <= thres] = 1
        object_ratio = 1 - mask.sum() / diff_sign.size(-1)
        return mask.unsqueeze(0), object_ratio
    
if __name__ == "__main__":
    clip_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(clip_id).to(device)

    object_style_model = OSModel(model, clip_id, "./mini_100.txt", device)
    object_mask, object_ratio = object_style_model.forward("a photo of orange", "orange")
    print("ratio of mask is: {}".format(object_ratio))