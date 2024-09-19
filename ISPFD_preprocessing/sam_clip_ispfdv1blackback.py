import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
from PIL import Image
import os
import numpy as np
from typing import Any, Dict, List
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel
import torch

parser = argparse.ArgumentParser(description=())
parser.add_argument("--parentdir", type=str)
parser.add_argument("--dstndir", type=str)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--convert-to-rle",action="store_true")
amg_settings = parser.add_argument_group("AMG Settings")
amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
)
amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)
amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)
amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)
amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)
amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)
amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)
amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)
amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)
amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)
amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def write_masks_to_folder(masks):
    masks_lst = list()
    box_lst = list()
    for _, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masks_lst.append(mask * 255)
        box_lst.append(mask_data['bbox'])
    return masks_lst, box_lst

def pad_and_crop_mask(mask, image, padding):
  non_zero_indices = np.where(mask == 255)
  y_min, y_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0]) + 1
  x_min, x_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1]) + 1
  pad_width = ((padding, padding), (padding, padding))
  y_min = max(y_min - pad_width[0][0], 0)
  y_max = min(y_max + pad_width[0][1], image.shape[0])
  x_min = max(x_min - pad_width[1][0], 0)
  x_max = min(x_max + pad_width[1][1], image.shape[1])
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  cropped_image = image_rgb[y_min:y_max, x_min:x_max]
  h,w,_ = cropped_image.shape
  if h > w:
    cropped_image = cropped_image[:int((3*h)/4), :]
  else:
    cropped_image = cropped_image[:, :int(w/2)]
  return cropped_image

def get_object_from_mask(image, mask):
  if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
    raise TypeError("Image and mask must be NumPy arrays.")
  if image.shape[:2] != mask.shape:
    raise ValueError("Image and mask must have the same spatial dimensions.")
  object_image = np.zeros_like(image)
  object_image[mask == 255] = image[mask == 255]
  object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
  return object_image

def orient_and_adjust(image,bbox):
    if image.shape[1]>image.shape[0]: # -- image is horizontal
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # new_x = image.shape[1] - bbox[0] - bbox[2]
        # new_y = image.shape[0] - bbox[1] - bbox[3]
        # bbox = (new_x, new_y, bbox[2], bbox[3])
        # img = cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)
        # cv2.imwrite('test.jpg',image)
        box_mid = bbox[0] + (bbox[2]//2)
        if image.shape[1]//2 < box_mid:   #-----coming from left
            image = cv2.flip(image,0)
        else:   #-----coming from right
            image = cv2.rotate(image, cv2.ROTATE_180)
            image = cv2.flip(image,0)
        return "H",image
    else:
        # cv2.imwrite('test.jpg',image)
        box_mid = bbox[1] + (bbox[3]//2)
        if image.shape[0]//2 > box_mid: # ----- coming from down
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.flip(image,1)
        else:  # coming from up
           image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
           image = cv2.flip(image,1)
        return "V",image

def tight_crop_with_padding(image, padding=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x, y, w, h = x - padding, y - padding, w + padding * 2, h + padding * 2
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def split_image_vertically(image):
  height, width, channels = image.shape
  half_width = int(0.55*width)
  left_half = image[:, :half_width, :]
  return left_half

def main(args: argparse.Namespace):
    print("Loading model...")
    sam = sam_model_registry['vit_h'](checkpoint="< Path to sam_vit_h_4b8939.pth cloned from SAM v1 repo >").to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device=args.device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_prompt = "Human Finger"

    parent_folder = args.parentdir
    dstn_folder = args.dstndir

    targets = list()
    for file in os.listdir(parent_folder):
       targets.append(os.path.join(parent_folder,file))

    exce = list()
    for t in tqdm(targets):
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)
        dstn_file = t.split("/")[-1]
        count=1
        img_lst = list()
        sim_lst = list()
        if output_mode == "binary_mask":
            lst,box_lst = write_masks_to_folder(masks)
            for i in lst:
                i = get_object_from_mask(image, i)
                img = Image.fromarray(i)
                inputs = processor(text=[text_prompt], images=img, return_tensors="pt", padding=True).to(device=args.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                sim_lst.append(logits_per_image.cpu().numpy()[0])
                img_lst.append(i)
                count += 1
            best_image = img_lst[sim_lst.index(max(sim_lst))]
            bbox = box_lst[sim_lst.index(max(sim_lst))]
            # postprocessing
            orienta,best_image = orient_and_adjust(best_image,bbox)
            best_image = tight_crop_with_padding(best_image,5)
            best_image = split_image_vertically(best_image)
            try:
               cv2.imwrite(os.path.join(dstn_folder,t.split("/")[-1]),best_image)
            except:
               exce.append(t.split("/")[-1])
    print(f"number of files skipped: {len(exce)}")
    with open(dstn_folder.split("/")[-2]+"_"+dstn_folder.split("/")[-1]+"_exceptions.json",'w') as js:
       json.dump(exce)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)