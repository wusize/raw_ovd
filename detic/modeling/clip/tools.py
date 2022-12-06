import torch
import torch.nn.functional as F
from tqdm import tqdm
import detic.modeling.clip as CLIP
from PIL import Image
import os
from glob import glob
multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'a single {} in the scene.',
    'a single {}.',

    'a photo of {article} {}.',
    'a photo of the {}.',
    'a photo of one {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',

]

def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'
def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res


def build_text_embedding_lvis(categories, model):
    templates = multiple_templates

    with torch.no_grad():
        all_text_embeddings = []
        for category in tqdm(categories):
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = CLIP.tokenize(texts).cuda()  # tokenize

            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()

            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=0)

    return all_text_embeddings


def get_reference_feature(ref_text, clip_model, clip_preprocess):
    ref_image_path, ref_word = ref_text.split(";")
    reference_features = []
    if os.path.exists(ref_image_path):
        reference_features.append(get_image_feature(ref_image_path, clip_model, clip_preprocess))  # 1xE
    if ref_word != '':
        reference_features.append(build_text_embedding_lvis([ref_text], clip_model).float())
    assert len(reference_features) > 0
    reference_feature = torch.cat(reference_features, dim=0).mean(dim=0, keepdim=True)

    return F.normalize(reference_feature, dim=-1)


def get_image_feature(ref_image_path, clip_model, clip_preprocess):
    if os.path.isfile(ref_image_path):
        image = clip_preprocess(Image.open(ref_image_path)
                                ).unsqueeze(0)
        device = clip_model.positional_embedding.data.device
        image_feature = clip_model.encode_image(image.to(device), normalize=True).float()    # 1xE

    else:
        image_files = glob(f'{ref_image_path}/*')
        images = []
        for image_file in image_files:
            image = clip_preprocess(Image.open(image_file)
                                    ).unsqueeze(0)
            images.append(image)
        device = clip_model.positional_embedding.data.device
        image_features = clip_model.encode_image(torch.cat(images, dim=0).to(device), normalize=True).float()  # 1xE
        image_feature = image_features.mean(dim=0, keepdim=True)
        image_feature = F.normalize(image_feature, dim=-1)

    return image_feature
