import os


def process_proposals(batched_inputs, images, proposals):
    image_proposals = []
    for input, real_image_size, p in zip(batched_inputs, images.image_sizes, proposals):
        output = dict(file_name=os.path.basename(input['file_name']),
                      height=input['height'],
                      width=input['width'],
                      image_id=input['image_id'],
                      image_size=real_image_size,
                      proposals=p.proposal_boxes.tensor.cpu().numpy().tolist(),
                      objectness_scores=p.objectness_logits.sigmoid().cpu().numpy().tolist())
        image_proposals.append(output)
    return image_proposals
