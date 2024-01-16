import torch
import argparse
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from classification import ClassificationModel
import os, json
import time
from typing import Any, Mapping
import numpy as np
import random
import pdb


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def save_image(image, path):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    image.save(path)


def metric(probs, gt, return_false=False):
    bs = probs.size(0)
    max_v, max_index = torch.max(probs, dim=-1)
    acc = (max_index == gt).sum()
    if return_false:
        # pdb.set_trace()
        false_index = torch.where(max_index != gt)[0]
        return acc, false_index
    return acc


def write_log(file, text):
    file.write(text + "\n")
    print(text)

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='experiment configuration')

    args = argparse.Namespace()
    args.__dict__.update(read_json(parser.parse_args().config_path))
    print("======  Current Parameter =======")
    for para in args.__dict__:
        print(para + ': ' + str(args.__dict__[para]))

    # define the image output dir
    args.image_output_dir = os.path.join(r"../diffusion_outputs/attack/formal_experiment/",
                                         args.output_dir.split('formal_experiment/')[-1])
    print("generated path:{}".format(args.image_output_dir))
    if os.path.exists(args.image_output_dir):
        replace_type = input("The image output path has existed, replace all? (yes/no) ")
        if replace_type == "no":
            exit()
        elif replace_type == "yes":
            pass
        else:
            raise ValueError("Answer must be yes or no")
    os.makedirs(args.image_output_dir, exist_ok=True)

    # load attack prompt
    label_infos = ["oil painting", "watercolor", "sketch", "animation", "photorealistic"]

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # stabilityai/stable-diffusion-2-1-base  # runwayml/stable-diffusion-v1-5
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    pipe = pipe.to(device)
    image_length = 512

    # load style classification model
    style_classify_model = ClassificationModel(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                               label_txt=label_infos, device=device, mode='style')

    # load original prompt
    with open(args.prompt_path, 'r') as f:
        ori_prompts = f.readlines()

    # load the prompt label
    object_path = "./mini_100.txt"
    with open(object_path, 'r') as f:
        object_infos = f.readlines()

    # load object classification model
    object_classify_model = ClassificationModel(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                                label_txt=object_infos, device=device, mode='object')
    total_5_acc_style, total_10_acc_style, total_acc_style = 0, 0, 0
    total_5_acc_obj, total_10_acc_obj, total_acc_obj = 0, 0, 0
    gen_num = 10
    batch_size = 5
    batch = int(np.ceil(gen_num / batch_size))
    attack_goal_num = 0
    for style_goal in os.listdir(args.output_dir):
        print("\n Start to generate images of {}\n".format(style_goal))
        style_result_path = os.path.join(args.output_dir, style_goal)
        if not os.path.isdir(style_result_path):
            continue
        attack_goal_num += 1
        attack_path = os.path.join(style_result_path, "results.txt")
        with open(attack_path, 'r') as f:
            attack_infos = f.readlines()

        cur_style_output_dir = os.path.join(args.image_output_dir, style_goal)
        if os.path.exists(cur_style_output_dir):
            replace_type = input("The output acc path has existed!!! replace all? (yes/no) ")
            if replace_type == "no":
                exit()
            elif replace_type == "yes":
                pass
            else:
                raise ValueError("Answer must be yes or no")
        os.makedirs(cur_style_output_dir, exist_ok=True)
        output_file = open(os.path.join(cur_style_output_dir, "results.txt"), "w")

        # load style goal
        if style_goal in style_classify_model.labels:
            style_label = style_classify_model.labels.index(style_goal)
        else:
            style_classify_model.add_label_param([style_goal])
            style_label = len(style_classify_model.labels)

        # generate images
        each_5_acc_style, each_10_acc_style, each_acc_style = 0, 0, 0
        each_5_acc_obj, each_10_acc_obj, each_acc_obj = 0, 0, 0
        init_time = time.time()
        for i in range(1, len(attack_infos)):
            # set_random_seed(666)  # fix the seed between the original image and the adversarial image
            # ori_prompt = ori_prompts[i-1]
            # write_log(output_file, f"Generate {i-1}^th ori_prompt: {ori_prompt}")
            # for j in range(batch):
            #     num_images = min(100, (j+1)*batch_size) - j*batch_size
            #     guidance_scale = 9
            #     num_inference_steps = 25
            #     images = pipe(
            #         ori_prompt,
            #         num_images_per_prompt=num_images,
            #         guidance_scale=guidance_scale,
            #         num_inference_steps=num_inference_steps,
            #         height=image_length,
            #         width=image_length,
            #     ).images
            #     for img_num in range(num_images):
            #         save_image(images[img_num],
            #                    os.path.join(args.image_output_dir, f"original/{ori_prompt}/{img_num + j * 10}.png"))

            set_random_seed(666)  # fix the seed between the original image and the adversarial image
            prompt = attack_infos[i].strip()
            object_label = i - 1
            tar_object = object_infos[object_label].strip().lower()
            assert tar_object.replace("-", " - ") in prompt, "The adversarial prompt don't contain the original object," \
                                                         f"current object: {tar_object}, current prompt: {prompt}"
            cur_5_acc_style, cur_10_acc_style, cur_avg_acc_style = 0, 0, 0
            cur_5_acc_obj, cur_10_acc_obj, cur_avg_acc_obj = 0, 0, 0
            start_time = time.time()
            write_log(output_file, f"Generate {i}^th adv prompt: {prompt}, label: {tar_object}, attack: {style_goal}")
            for j in range(batch):
                num_images = min(gen_num, (j+1)*batch_size) - j*batch_size
                guidance_scale = 9
                num_inference_steps = 25
                images = pipe(
                    prompt,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=image_length,
                    width=image_length,
                ).images

                # style acc
                probs_style = style_classify_model.forward(images)
                acc_num = metric(probs_style, style_label)
                cur_avg_acc_style += acc_num
                if (j+1) * batch_size <= 5 and acc_num > 0:
                    cur_5_acc_style = 1
                if acc_num > 0:
                    cur_10_acc_style = 1

                # obj acc
                probs_obj = object_classify_model.forward(images)
                acc_num = metric(probs_obj, object_label)
                cur_avg_acc_obj += acc_num
                if (j+1) * batch_size <= 5 and acc_num > 0:
                    cur_5_acc_obj = 1
                if acc_num > 0:
                    cur_10_acc_obj = 1

                for img_num in range(num_images):
                    sign = probs_style[img_num].argmax(0) == style_label
                    sign &= probs_obj[img_num].argmax(0) == object_label
                    save_image(images[img_num],
                               os.path.join(cur_style_output_dir,
                                f"{tar_object}/{prompt}/{sign}_{img_num + j * batch_size}.png"))
            end_time = time.time()
            each_5_acc_style += cur_5_acc_style
            each_10_acc_style += cur_10_acc_style
            each_acc_style += cur_avg_acc_style
            each_5_acc_obj += cur_5_acc_obj
            each_10_acc_obj += cur_10_acc_obj
            each_acc_obj += cur_avg_acc_obj
            write_log(output_file,
                      f"{prompt} 5_acc_style is: {cur_5_acc_style}, 10_acc_style is {cur_10_acc_style}, "
                      f"avg_acc_style is {cur_avg_acc_style}")
            write_log(output_file,
                      f"{prompt} 5_acc_obj is: {cur_5_acc_obj}, 10_acc_obj is {cur_10_acc_obj}, "
                      f"avg_acc_obj is {cur_avg_acc_obj}")
            write_log(output_file, "Spent time: {:.3f}s".format(end_time - start_time))
        write_log(output_file, "\nEnd the testing stage\n")
        write_log(output_file, "style acc-5 is {:.3f}%".format(each_5_acc_style * 100 / (len(attack_infos) - 1)))
        write_log(output_file, "style acc-10 is {:.3f}%".format(each_10_acc_style * 100 / (len(attack_infos) - 1)))
        write_log(output_file, "style acc is {:.3f}%".format(each_acc_style * 100 / gen_num / (len(attack_infos) - 1)))
        write_log(output_file, "obj acc-5 is {:.3f}%".format(each_5_acc_obj * 100 / (len(attack_infos) - 1)))
        write_log(output_file, "obj acc-10 is {:.3f}%".format(each_10_acc_obj * 100 / (len(attack_infos) - 1)))
        write_log(output_file, "obj acc is {:.3f}%".format(each_acc_obj * 100 / gen_num / (len(attack_infos) - 1)))
        finish_time = time.time()
        all_time = finish_time - init_time
        write_log(output_file,
                  "spent time is {}h{}m\n".format(
                      all_time // 3600, (all_time % 3600) // 60))
        total_5_acc_style += each_5_acc_style * 100 / (len(attack_infos) - 1)
        total_10_acc_style += each_10_acc_style * 100 / (len(attack_infos) - 1)
        total_acc_style += each_acc_style * 100 / gen_num / (len(attack_infos) - 1)
        total_5_acc_obj += each_5_acc_obj * 100 / (len(attack_infos) - 1)
        total_10_acc_obj += each_10_acc_obj * 100 / (len(attack_infos) - 1)
        total_acc_obj += each_acc_obj * 100 / gen_num / (len(attack_infos) - 1)
    total_result_path = os.path.join(args.image_output_dir, "results.txt")
    output_file = open(total_result_path, "w")
    write_log(output_file, "Finish All Testing")
    write_log(output_file, "Final style acc-5  is {:.3f}%".format(total_5_acc_style / attack_goal_num))
    write_log(output_file, "Final style acc-10 is {:.3f}%".format(total_10_acc_style / attack_goal_num))
    write_log(output_file, "Final style acc is {:.3f}%".format(total_acc_style / attack_goal_num))
    write_log(output_file, "Final obj acc-5 is {:.3f}%".format(total_5_acc_obj / attack_goal_num))
    write_log(output_file, "Final obj acc-10 is {:.3f}%".format(total_10_acc_obj / attack_goal_num))
    write_log(output_file, "Final obj acc is {:.3f}%".format(total_acc_obj / attack_goal_num))