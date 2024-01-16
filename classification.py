import torch
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import os, json


class ClassificationModel(nn.Module):
    def __init__(self, model_id, label_txt=None, device='cpu', mode='style'):
        super(ClassificationModel, self).__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.image_processor = AutoProcessor.from_pretrained(model_id)
        self.Softmax = nn.Softmax(dim=-1)
        self.mode = mode
        self.init_fc_param(label_txt)
        self.fix_backbobe()
        self.to(self.device)

    def init_fc_param(self, label_path=None):
        if label_path is None:
            raise ValueError('Please input the path of ImageNet1K annotations')
        if type(label_path) == str:
            with open(label_path, 'r') as f:
                infos = f.readlines()
        else:
            infos = label_path
        prompts = []
        labels = []
        for cla in infos:
            labels.append(cla.lower().strip())
            if self.mode == "style":
                prompts.append(f"a photo with the {cla.lower().strip()} style")  #
            elif self.mode == 'object':
                prompts.append(f"a photo of {cla.lower().strip()}")
            else:
                raise ValueError('Please supply the classification mode')
        # pdb.set_trace()
        with torch.no_grad():
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        self.text_embeds = text_embeds.to(self.device)
        self.labels = labels

    def add_label_param(self, label: list):
        if self.mode == 'style':
            inputs = [f"a photo with the {ll} style" for ll in label]
        else:
            inputs = [f"a photo of {ll}" for ll in label]
        with torch.no_grad():
            inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds.to(self.device)
        self.text_embeds = torch.cat((self.text_embeds, text_embeds), dim=0)
        self.labels.extend(ll.lower().strip() for ll in label)

    def get_scores(self, image_embeds):
        logit_scale = self.model.logit_scale.exp()
        logit_scale = logit_scale.to(self.device)
        logits_per_image = torch.matmul(image_embeds, self.text_embeds.t()) * logit_scale
        return logits_per_image

    def forward(self, image):
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt")
            # pdb.set_trace()
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.device)
            image_embeds = self.model.get_image_features(**inputs)
            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            scores = self.get_scores(image_embeds)
            probs = self.Softmax(scores)
        return probs

    def fix_backbobe(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def get_params(self):
        params = []
        params.append({'params': self.model.parameters()})
        return params

    def to(self, device):
        self.model = self.model.to(device)