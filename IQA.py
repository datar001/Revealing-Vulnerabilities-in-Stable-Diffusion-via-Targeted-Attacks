import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import ToTensor
import argparse
import os
from PIL import Image
from shutil import move
import tqdm
from scipy import linalg
from test_style import metric
from classification import ClassificationModel
import numpy as np
import scipy
import pdb
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.dropout = inception.dropout
        self.fc = inception.fc

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        flatten = torch.flatten(self.dropout(x), 1)
        preds = self.fc(flatten)
        return x.view(x.size(0), -1), preds


def inception_score(images, batch_size=100, splits=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(weights=True, transform_input=False).to(device)
    model.eval()

    scores = []
    entropys = []
    num_batches = len(images) // batch_size

    with torch.no_grad():
        for i in tqdm.tqdm(range(num_batches)):
            batch = torch.stack([ToTensor()(img).to(device) for img in images[i * batch_size:(i + 1) * batch_size]])
            preds = nn.Softmax(dim=1)(model(batch))
            p_yx = preds.log()
            p_y = preds.mean(dim=0).log()
            entropy = torch.sum(preds * p_yx, dim=1).mean()
            kl_divergence = torch.sum(preds * (p_yx - p_y), dim=1).mean()
            entropys.append(torch.exp(entropy))
            scores.append(torch.exp(kl_divergence))
    entropys = torch.stack(entropys)
    mean_entropys = entropys.mean()
    std_entropys = entropys.std(dim=-1)

    scores = torch.stack(scores)
    mean_score = scores.mean()
    std_score = scores.std(dim=-1)
    return mean_score.item(), std_score.item(), mean_entropys, std_entropys


def calculate_activation_statistics(images, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    act_values = []

    with torch.no_grad():
        for img in tqdm.tqdm(images):
            img = ToTensor()(img).unsqueeze(0).to(device)
            act = model(img).detach().cpu()
            act_values.append(act)

    act_values = torch.cat(act_values, dim=0).detach().numpy()
    mu = np.mean(act_values, axis=0)
    sigma = np.cov(act_values, rowvar=False)

    return mu, sigma


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)


def frechet_inception_distance(real_images, generated_images, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3().to(device)
    model.eval()
    print("get features of the real images")
    mu_real, sigma_real = calculate_activation_statistics(real_images, model)
    print("get features of the generated images")
    mu_fake, sigma_fake = calculate_activation_statistics(generated_images, model)

    fid_score = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score.item()


def get_model_outputs(model, images, batch_size=100):
    preds, act_values = [], []
    num_batches = len(images) // batch_size
    assert num_batches * batch_size == len(images)
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_batches)):
            batch = torch.stack([ToTensor()(img).to(device) for img in images[i * batch_size:(i + 1) * batch_size]])
            act, pred = model(batch)
            pred = nn.Softmax(dim=1)(pred)
            act_values.append(act)
            preds.append(pred)

    act_values = torch.stack(act_values, dim=0).view(-1, act.size(-1))
    preds = torch.cat(preds, dim=0).view(-1, pred.size(-1))

    return act_values.cpu().numpy(), preds.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_img_path', type=str, required=True, help='the path of generated images')
    parser.add_argument('--task', type=str, default='object', help='object task or style task')
    parser.add_argument('--attack_goal_path', type=str, default=None, help='the path of referenced images')
    parser.add_argument('--metric', type=str, default='image_quality', help='[image_quality, attack_acc]')
    parser.add_argument('--depth', type=int, default=4, help='3 or 4 for the dir depth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3().to(device)
    model.eval()

    gen_imgs = []
    gen_img_ori_labels = []
    gen_img_goal_labels = []
    if args.depth == 4:
        for goal_dir in os.listdir(args.gen_img_path):
            goal_path = os.path.join(args.gen_img_path, goal_dir)
            if not os.path.isdir(goal_path):
                continue
            image_num_per_goal = 0
            if os.path.isdir(goal_path):
                for prompt_dir in os.listdir(goal_path):
                    cur_path = os.path.join(goal_path, prompt_dir)
                    if os.path.isdir(cur_path):
                        for adv_prompt in os.listdir(cur_path):
                            img_dir = os.path.join(cur_path, adv_prompt)
                            for img_name in os.listdir(img_dir):
                                assert img_name.endswith(".png")
                                gen_imgs.append(Image.open(os.path.join(img_dir, img_name)))
                                gen_img_ori_labels.append(prompt_dir.split("_")[0])
                                gen_img_goal_labels.append(goal_dir)
                                image_num_per_goal += 1

                print(f"{goal_dir} goal has {image_num_per_goal} images")
    elif args.depth == 3:
        for prompt_dir in os.listdir(args.gen_img_path):
            cur_path = os.path.join(args.gen_img_path, prompt_dir)
            if os.path.isdir(cur_path):
                for adv_prompt in os.listdir(cur_path):
                    if prompt_dir.replace("-", "_-_").replace(" ", "_").lower() in adv_prompt:
                        img_dir = os.path.join(cur_path, adv_prompt)
                        print(f"{prompt_dir} has {len(os.listdir(img_dir))} images")
                        for img_name in os.listdir(img_dir):
                            assert img_name.endswith(".png")
                            gen_imgs.append(Image.open(os.path.join(img_dir, img_name)))
                            gen_img_ori_labels.append(prompt_dir)
                            gen_img_goal_labels.append(args.gen_img_path.split("/")[-1])
                    else:
                        pdb.set_trace()
    print(f"Total has {len(gen_imgs)} generated images")

    if args.metric == "image_quality":
        gen_images_num_per_goal = 100 * 10  # 100 prompt, 10 images per prompt
        goal_num = len(gen_imgs) // gen_images_num_per_goal
        with open(args.attack_goal_path, "r") as f:
            real_img_path = f.readlines()
            goals = [goal_path.split("/")[-1].lower().strip() for goal_path in real_img_path]
        assert goal_num == len(real_img_path), "The goal num of generate images do not equal to the category number " \
                                               f"of target images, but get {goal_num} goal num and {len(real_img_path)} " \
                                               f"category num"
        total_IS, total_FID = [], []
        for i in range(goal_num):
            gen_images_per_goal = gen_imgs[i * gen_images_num_per_goal:(i + 1) * gen_images_num_per_goal]
            gen_images_goal = gen_img_goal_labels[i * gen_images_num_per_goal:(i + 1) * gen_images_num_per_goal]
            assert len(
                set(gen_images_goal)) == 1, f"multi goals happened in computing FID score, {set(gen_images_goal)}"
            goal = gen_images_goal[0]
            goal_img_path = real_img_path[goals.index(goal)].strip()
            real_imgs = []
            for img_name in os.listdir(goal_img_path):
                img_path = os.path.join(goal_img_path, img_name)
                real_imgs.append(Image.open(img_path))
            print(f"Total has {len(real_imgs)} real images in {goal} goal")

            gen_acts, gen_preds = get_model_outputs(model, gen_images_per_goal, batch_size=10)
            mu_gen = np.mean(gen_acts, axis=0)
            sigma_gen = np.cov(gen_acts, rowvar=False)

            ## IS score
            IS_batch_size = 10
            IS_batch = len(gen_images_per_goal) // IS_batch_size
            split_entropys, split_scores = [], []
            for i in range(IS_batch):
                cur_preds = gen_preds[i * IS_batch_size: min((i + 1) * IS_batch_size, len(gen_images_per_goal))]
                py = np.mean(cur_preds, axis=0)
                scores, entropys = [], []
                for j in range(cur_preds.shape[0]):
                    pyx = cur_preds[j, :]
                    scores.append(scipy.stats.entropy(pyx, py))
                    entropys.append(scipy.stats.entropy(pyx))
                split_scores.append(np.exp(np.mean(scores)))
                split_entropys.append(np.exp(np.mean(entropys)))

            mean_entropys = np.mean(split_entropys)
            std_entropys = np.std(split_entropys)

            mean_score = np.mean(split_scores)
            std_score = np.std(split_scores)
            print(f"{goal} goal: Entropy: {mean_entropys}, IS: {mean_score}")
            total_IS.append(mean_score)

            # FID
            real_acts, real_preds = get_model_outputs(model, real_imgs, batch_size=len(real_imgs))
            mu_real = np.mean(real_acts, axis=0)
            sigma_real = np.cov(real_acts, rowvar=False)
            fid_score = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
            print(f"{goal} goal: FID: {fid_score.item()}")
            total_FID.append(fid_score.item())
        print(f"Mean IS: {sum(total_IS) / goal_num}, Mean FID: {sum(total_FID) / goal_num}")

    elif args.metric == "attack_acc":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.task == "object":
            label_path = r"./mini_100.txt"
            with open(label_path, "r") as f:
                label_infos = f.readlines()
                label_infos = [label.lower().strip() for label in label_infos]
        elif args.task == "style":
            # load style label
            label_infos = ["oil painting", "watercolor", "sketch", "animation", "photorealistic"]
            # load style classification model
            style_classify_model = ClassificationModel(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                                       label_txt=label_infos, device=device, mode=args.task)
        else:
            raise ValueError("Task must be object or task")

        # load the prompt label
        object_path = "./mini_100.txt"
        with open(object_path, 'r') as f:
            object_infos = f.readlines()
            object_infos = [obj.lower().strip() for obj in object_infos]
        # load object classification model
        object_classify_model = ClassificationModel(model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                                    label_txt=object_infos, device=device, mode='object')
        with open(args.attack_goal_path, "r") as f:
            attack_goals = f.readlines()
            attack_goals = [goal.split("/")[-1].lower().strip() for goal in attack_goals]

        if args.task == "style":
            for goal in attack_goals:
                if goal not in style_classify_model.labels:
                    style_classify_model.add_label_param([goal.lower()])
            object_goal = None
        else:
            for goal in attack_goals:
                if goal not in object_classify_model.labels:
                    object_classify_model.add_label_param([goal.lower()])

        target_goal_num = len(attack_goals)
        image_num_per_goal = len(gen_imgs) // target_goal_num
        image_num_per_prompt = 10
        prompt_num = len(gen_imgs) // image_num_per_prompt // target_goal_num
        split_num_per_prmpt = 5
        batch_num_per_prompt = image_num_per_prompt // split_num_per_prmpt

        total_5_acc_style, total_10_acc_style, total_acc_style = 0, 0, 0
        total_5_acc_obj, total_10_acc_obj, total_acc_obj = 0, 0, 0
        for k, goal in enumerate(attack_goals):
            images_per_goal = gen_imgs[k * image_num_per_goal: (k + 1) * image_num_per_goal]
            labels_per_goal = gen_img_ori_labels[k * image_num_per_goal: (k + 1) * image_num_per_goal]
            each_5_acc_style, each_10_acc_style, each_acc_style = 0, 0, 0
            each_5_acc_obj, each_10_acc_obj, each_acc_obj = 0, 0, 0
            for i in range(prompt_num):
                images_per_prompt = images_per_goal[i * image_num_per_prompt: (i + 1) * image_num_per_prompt]
                labels_per_prompt = labels_per_goal[i * image_num_per_prompt: (i + 1) * image_num_per_prompt]
                if args.task == "style":
                    assert len(set(labels_per_prompt)) == 1, "one prompt contains multi-class images!! Attention!"
                    object_label = object_classify_model.labels.index(labels_per_prompt[0].lower())
                    style_label = style_classify_model.labels.index(goal.lower())
                else:
                    object_label = object_classify_model.labels.index(goal.lower())

                cur_5_acc_style, cur_10_acc_style, avg_acc_style = 0, 0, 0
                cur_5_acc_obj, cur_10_acc_obj, avg_acc_obj = 0, 0, 0
                for j in range(batch_num_per_prompt):
                    images_per_batch = images_per_prompt[j * split_num_per_prmpt: (j + 1) * split_num_per_prmpt]

                    # style acc
                    if args.task == "style":
                        probs_style = style_classify_model(images_per_batch)
                        acc_num = metric(probs_style, style_label)
                        if (j + 1) * split_num_per_prmpt <= 5 and acc_num > 0:
                            cur_5_acc_style = 1
                        if acc_num > 0:
                            cur_10_acc_style = 1
                        avg_acc_style += acc_num

                    # obj acc
                    probs_obj = object_classify_model(images_per_batch)
                    acc_num = metric(probs_obj, object_label)
                    avg_acc_obj += acc_num

                    if (j + 1) * split_num_per_prmpt <= 5 and acc_num > 0:
                        cur_5_acc_obj = 1
                    if acc_num > 0:
                        cur_10_acc_obj = 1

                each_5_acc_obj += cur_5_acc_obj
                each_10_acc_obj += cur_10_acc_obj
                each_acc_obj += avg_acc_obj
                print(f"{i}^th 5_acc_obj is: {cur_5_acc_obj}, 10_acc_obj is {cur_10_acc_obj}, "
                      f"avg_acc_obj is {avg_acc_obj}")
                if args.task == "style":
                    each_5_acc_style += cur_5_acc_style
                    each_10_acc_style += cur_10_acc_style
                    each_acc_style += avg_acc_style
                    print(f"{i}^th 5_acc_style is: {cur_5_acc_style}, 10_acc_style is {cur_10_acc_style}, "
                          f"avg_acc_style is {avg_acc_style}")
            if args.task == "style":
                print("Each 5 acc style is {:.3f}%".format(each_5_acc_style * 100 / prompt_num))
                print("Each 10 acc style is {:.3f}%".format(each_10_acc_style * 100 / prompt_num))
                print("Each acc style is {:.3f}%".format(each_acc_style * 100 / image_num_per_prompt / prompt_num))
            print("Each 5 acc obj is {:.3f}%".format(each_5_acc_obj * 100 / prompt_num))
            print("Each 10 acc obj is {:.3f}%".format(each_10_acc_obj * 100 / prompt_num))
            print("Each acc obj is {:.3f}%".format(each_acc_obj * 100 / image_num_per_prompt / prompt_num))
            if args.task == "style":
                total_5_acc_style += each_5_acc_style * 100 / prompt_num
                total_10_acc_style += each_10_acc_style * 100 / prompt_num
                total_acc_style += each_acc_style * 100 / image_num_per_prompt / prompt_num
            total_5_acc_obj += each_5_acc_obj * 100 / prompt_num
            total_10_acc_obj += each_10_acc_obj * 100 / prompt_num
            total_acc_obj += each_acc_obj * 100 / image_num_per_prompt / prompt_num
        print("Final 5 acc style is {:.3f}%".format(total_5_acc_style / target_goal_num))
        print("Final 10 acc style is {:.3f}%".format(total_10_acc_style / target_goal_num))
        print("Total acc style is {:.3f}%".format(total_acc_style / target_goal_num))
        print("Final 5 acc obj is {:.3f}%".format(total_5_acc_obj / target_goal_num))
        print("Final 10 acc obj is {:.3f}%".format(total_10_acc_obj / target_goal_num))
        print("Total acc obj is {:.3f}%".format(total_acc_obj / target_goal_num))
    else:
        raise ValueError("metric must be {image_quality, attack_acc}")