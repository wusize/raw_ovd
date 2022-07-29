from .context_modelling import ContextModelling, pseudo_permutations
import torch
from timm.loss import SoftTargetCrossEntropy
from .utils import bbox_xyxy_to_cxcywh
import numpy as np
import math
import random
from torch.cuda.amp import autocast


# TODO: support caption
class UCLContextModelling(ContextModelling):
    def __init__(self, *args, **kwargs):
        super(UCLContextModelling, self).__init__(*args, **kwargs)
        self.num_classes = self.cfg.NUM_CLASSES
        class_embeddings = self.cfg.CLASS_EMBEDDINGS
        if class_embeddings != '':
            class_embeddings = torch.tensor(
                np.load(class_embeddings), dtype=torch.float32)
            self.register_buffer('class_embeddings', class_embeddings, persistent=False)

        self.caption_loss = SoftTargetCrossEntropy()

    def caption_contrast(self, caption_normed_boxes, predictions, clip_model, image_info):
        clip_model.eval()
        batch_size = len(caption_normed_boxes)
        caption_pseudo_words = predictions.pop('caption_pseudo_words')
        device = caption_pseudo_words.device
        all_clip_caption_features, num_captions_per_image = self.get_caption_features([v['captions']
                                                                                       for v in image_info.values()],
                                                                                      device,
                                                                                      clip_model)

        num_boxes_per_image = [b.shape[0] for b in caption_normed_boxes]
        positions = bbox_xyxy_to_cxcywh(torch.cat(caption_normed_boxes, dim=0))
        position_embeddings = self.positional_embed(positions)
        permutations_per_image = [pseudo_permutations(n, min(math.factorial(n),
                                                             self.caption_cfg.MAX_PERMUTATIONS))
                                  for n in num_boxes_per_image]
        if self.caption_cfg.ADD_IMAGE_BOX:
            position_embeddings = position_embeddings.split(num_boxes_per_image, dim=0)
            zero_pes = [torch.zeros_like(position_embeddings[0][:1]) for _ in caption_normed_boxes]
            position_embeddings = torch.cat([torch.cat([pe, ze], dim=0)
                                             for pe, ze in zip(position_embeddings, zero_pes)], dim=0)
            num_boxes_per_image = [n+1 for n in num_boxes_per_image]
            permutations_per_image = [[perm + [len(perm)] for perm in b] for b in permutations_per_image]
        num_perms_per_image = [len(b) for b in permutations_per_image]
        caption_pseudo_words = (caption_pseudo_words + position_embeddings).split(num_boxes_per_image, dim=0)
        caption_pseudo_words = [[ws[perm] for perm in perms]
                                for ws, perms in zip(caption_pseudo_words, permutations_per_image)]
        caption_pseudo_sequences = [perm for b in caption_pseudo_words for perm in b]
        words_split = [perm.shape[0] for b in caption_pseudo_words for perm in b]
        words_mask = self._drop_word(torch.cat(caption_pseudo_sequences, dim=0)).split(words_split,
                                                                                       dim=0)
        caption_pseudo_sequences = [seq.flatten(0, 1)[wm.view(-1)]
                                    for seq, wm in zip(caption_pseudo_sequences,  words_mask)]
        context_length = max([seq.shape[0] for seq in caption_pseudo_sequences])
        with autocast():
            # TODO: get local image tokens
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text(
                caption_pseudo_sequences,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features = \
                clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                              text_pe=True, normalize=True,
                                              return_word_tokens=False)
            clip_text_features = clip_text_features.float()

        if all_clip_caption_features is None:
            caption_valid = torch.zeros(batch_size, device=device)
            clip_caption_features = torch.zeros(batch_size, 512, device=device)
            caption_img_ids = torch.tensor(list(image_info.keys()), device=device,
                                           dtype=torch.float32)
        else:
            caption_valid = []
            clip_caption_features = all_clip_caption_features.split(num_captions_per_image, dim=0)
            clip_caption_features_list = []
            caption_img_ids = []
            max_caps = self.caption_cfg.CAPS_PER_IMG
            for img_id, num_cap, cap_feat in zip(image_info.keys(),
                                                 num_captions_per_image, clip_caption_features):
                assert num_cap == cap_feat.shape[0]
                if num_cap > 0:
                    num_samples = min(max_caps, num_cap)
                    caption_valid.append(torch.ones(num_samples, device=device))
                    sampled_ids = random.sample(range(num_cap), num_samples)
                    clip_caption_features_list.append(cap_feat[sampled_ids])
                    caption_img_ids.append(img_id * torch.ones(num_samples, device=device))
                else:
                    clip_caption_features_list.append(torch.zeros(1, 512, device=device))
                    caption_valid.append(torch.zeros(1, device=device))
                    caption_img_ids.append(img_id * torch.ones(1, device=device))
            caption_valid = torch.cat(caption_valid)
            clip_caption_features = torch.cat(clip_caption_features_list)
            caption_img_ids = torch.cat(caption_img_ids)
        invalid_caps = torch.where(caption_valid < 1.0)[0]
        pred_image_ids = torch.tensor([k for num_perms, k in zip(num_perms_per_image,
                                                                 image_info.keys()) for _ in range(num_perms)],
                                      device=device)
        num_preds = clip_text_features.shape[0]
        assert sum(num_perms_per_image) == num_preds
        assert pred_image_ids.shape[0] == num_preds

        global_clip_text_features = self.queues.get_queue('clip_cap_text_features')  # add "_cap_" to avoid conflict
        contrast_clip_text_features = torch.cat([clip_text_features,
                                                 global_clip_text_features[..., :-1]], dim=0)
        contrast_clip_text_image_ids = torch.cat([pred_image_ids,
                                                  global_clip_text_features[..., -1]], dim=0)

        global_clip_caption_features = self.queues.get_queue('clip_caption_features')
        contrast_clip_caption_features = torch.cat([clip_caption_features,
                                                    global_clip_caption_features[..., :-1]], dim=0)
        contrast_clip_caption_image_ids = torch.cat([caption_img_ids,
                                                     global_clip_caption_features[..., -1]], dim=0)

        # matrix 0
        similarity_matrix_0 = self.ce_temp * contrast_clip_text_features @ contrast_clip_caption_features.T
        label_matrix_0 = (contrast_clip_text_image_ids[:, None] == contrast_clip_caption_image_ids[None]).float()

        # matrix 1
        similarity_matrix_1 = self.ce_temp * contrast_clip_caption_features @ contrast_clip_text_features.T
        label_matrix_1 = (contrast_clip_caption_image_ids[:, None] == contrast_clip_text_image_ids[None]).float()

        # mask invalid captions
        if len(invalid_caps) > 0:
            # matrix 0
            invalid_label_matrix_0 = label_matrix_0[:, invalid_caps]
            invalid_mask_0 = torch.where(invalid_label_matrix_0 > 0.0, self.ce_temp, -self.ce_temp)
            similarity_matrix_0[:, invalid_caps] = invalid_mask_0

            # matrix 1
            invalid_label_matrix_1 = label_matrix_1[invalid_caps]
            invalid_mask_1 = torch.where(invalid_label_matrix_1 > 0.0, self.ce_temp, -self.ce_temp)
            similarity_matrix_1[invalid_caps] = invalid_mask_1

        loss_0 = self.caption_loss(similarity_matrix_0, label_matrix_0)
        loss_1 = self.caption_loss(similarity_matrix_1, label_matrix_1)

        loss = loss_0 * 0.5 + loss_1 * 0.5

        if all_clip_caption_features is None:
            clip_caption_features_update = -torch.ones(1, 512 + 1, device=device)
        else:
            all_cap_image_ids = [img_id for img_id, num_cap in zip(image_info.keys(), num_captions_per_image)
                                 for _ in range(num_cap)]
            all_cap_image_ids = torch.tensor(all_cap_image_ids,
                                             device=device, dtype=torch.float32).view(-1, 1)
            clip_caption_features_update = torch.cat([all_clip_caption_features,
                                                      all_cap_image_ids], dim=-1)

        queue_update = dict(clip_caption_features=clip_caption_features_update,
                            clip_cap_text_features=torch.cat([clip_text_features,
                                                              pred_image_ids[:, None]], dim=-1))

        return dict(caption_loss=loss * self.cfg.CAPTION_LOSS_WEIGHT), queue_update
