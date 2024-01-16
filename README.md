# Revealing Vulnerabilities in Stable Diffusion via Targeted Attacks

<img src=examples/framework.png  width="70%" height="40%">

## Dependencies

- PyTorch == 2.0.1
- transformers == 4.23.1
- diffusers == 0.11.1
- ftfy==6.1.1
- accelerate=0.22.0
- python==3.8.13

## Usage

1. Download the [word2id.pkl and wordvec.pkl](https://drive.google.com/drive/folders/1tNa91aGf5Y9D0usBN1M-XxbJ8hR4x_Jq?usp=sharing) for the synonym model, and put download files into the Word2Vec dir.

2. A script is provided to perform targeted attacks for Stable Diffusion

```sh
# Traning for generating the adversarial prompts
python run.py --config_path ./object_config.json  # Object attacks
python run.py --config_path ./style_config.json  # Style attacks
# Testing for evaluating the attack success rate
python test_object_multi.py --config_path ./object_config.json  # Object attack 
python test_style_multi.py --config_path ./style_config.json # Style attack
# Testing for evaluating FID score of generated images
python IQA.py --gen_img_path [the root of generated images] --task [object or style] --attack_goal_path [the path of referenced images] --metric image_quality 
```

## Parameters

Config can be loaded from a JSON file. 

Config has the following parameters:

- `add_suffix_num`: the number of suffixes in the word addition perturbation strategy. The default is 5.
- `replace_type`: a list for specifying the word types in the word substitution strategy. The default is ['all'] that represent replace all words except the noun. Optional: ["verb", "adj", "adv", "prep"]
- `synonym_num`: The forbidden number of synonyms. The default is 10.
- `iter`: the total number of iterations. The default is 500.
- `lr`: the learning weight for the [optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html). The default is 0.1
- `weight_decay`: the weight decay for the optimizer.
- `loss_weight`: The weight of MSE loss in style attacks.
- `print_step`: The number of steps to print a line giving current status
- `batch_size`: number of referenced images used for each iteration.
- `clip_model`: the name of the CLiP model for use with . `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"` is the model used in SD 2.1.
- `prompt_path`: The path of clean prompt file.
- `task`: The targeted attack task. Optional: `"object"`or `"style"`
- `forbidden_words`: A txt file for representing the forbidden words for each target goal.
- `target_path`: The file path of referenced images.
- `output_dir`: The path for saving the learned adversarial prompts.

