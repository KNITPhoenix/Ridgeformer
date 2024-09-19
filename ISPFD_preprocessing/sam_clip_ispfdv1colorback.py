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

def calculate_total_zeros_in_stride_right(array, start_column, stride_length):
    end_column = start_column + stride_length
    columns_to_check = array[:, start_column:end_column]
    total_zeros = np.sum(columns_to_check == 0)
    return total_zeros

def calculate_total_zeros_in_left_stride(array, start_column, stride_length):
    end_column = max(0, start_column - stride_length)
    columns_to_check = array[:, end_column:start_column]
    total_zeros = np.sum(columns_to_check == 0)
    return total_zeros

def calculate_total_zeros_in_downward_stride(matrix, start_row, stride_length):
    end_row = min(start_row + stride_length, matrix.shape[0])
    rows_to_check = matrix[start_row:end_row, :]
    total_zeros = np.sum(rows_to_check == 0)
    return total_zeros

def calculate_total_zeros_in_upward_stride(matrix, start_row, stride_length):
    end_row = max(0, start_row - stride_length)
    rows_to_check = matrix[end_row:start_row, :]
    total_zeros = np.sum(rows_to_check == 0)
    return total_zeros

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
        img = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(img.shape[1]):
            if np.count_nonzero(img[:, i]) >= 20:
                left_index = i
                break
        for i in range(img.shape[1] - 1, -1, -1):
            if np.count_nonzero(img[:, i]) >= 20:
                right_index = i
                break
        total_zeros_towards_right = calculate_total_zeros_in_stride_right(img, left_index, 15)
        total_zeros_towards_left  = calculate_total_zeros_in_left_stride(img, right_index, 15)
        if total_zeros_towards_right > total_zeros_towards_left:  #---coming from left
            image = cv2.flip(image,0)
            orien = 'No'
        else:   #-----coming from right
            image = cv2.rotate(image, cv2.ROTATE_180)
            image = cv2.flip(image,0)
            orien = '180'
        return "H", image, orien
    else:
        img = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(img.shape[0]):
            if np.count_nonzero(img[i, :]) >= 20:
                top_index = i
                break
        for i in range(img.shape[0] - 1, -1, -1):
            if np.count_nonzero(img[i, :]) >= 20:
                bottom_index = i
                break
        total_zeros_towards_down = calculate_total_zeros_in_downward_stride(img, top_index, 15)
        total_zeros_towards_up   = calculate_total_zeros_in_upward_stride(img, bottom_index, 15)
        if total_zeros_towards_down > total_zeros_towards_up: # ----- coming from down
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.flip(image,0)
            orien = 'Rotate90anti'
        else:  # coming from up
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = cv2.flip(image,0)
            orien = 'Rotate90'
        return "V", image, orien

def tight_crop_with_padding(image, original_image, padding=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x, y, w, h = x - padding, y - padding, w + padding * 2, h + padding * 2
    cropped_image = image[y:y+h, x:x+w]
    crop_original = original_image[y:y+h, x:x+w]
    return cropped_image, crop_original

def split_image_vertically(image):
    height, width, channels = image.shape
    half_width = int(0.55*width)
    left_half = image[:, :half_width, :]
    return left_half

def tight_bounding_box(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(cnt)
    # print(ellipse[1][0]/ellipse[1][1])
    # cv2.imwrite("test.jpg",cv2.ellipse(image, ellipse, (0, 255, 0), 2))
    # exit(0)
    return ellipse[1][0]/ellipse[1][1]

def resize_image_with_aspect_ratio(image, max_width=None, max_height=None):
    height, width, _ = image.shape
    aspect_ratio = width / height
    if max_width and width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    elif max_height and height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    else:
        return image
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def main(args: argparse.Namespace):
    print("Loading model...")
    sam = sam_model_registry['vit_h'](checkpoint="< Path to sam_vit_h_4b8939.pth cloned from SAM v1 repo >").to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device=args.device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_prompt = "Human Finger"

    parent_folder = args.parentdir
    dstn_folder = args.dstndir

    targets = list()
    for file in os.listdir(parent_folder):
       targets.append(os.path.join(parent_folder,file))

    exce = list()
    skipper = 0
    for t in tqdm(targets):
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        # image = resize_image_with_aspect_ratio(image, 3264, 2448)
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
            try:
                orienta, best_image, orien = orient_and_adjust(best_image,bbox)
                if orienta == 'V' and orien == 'Rotate90':
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif orienta == 'V' and orien == 'Rotate90anti':
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif orienta == 'H' and orien == 'No':
                    pass
                elif orienta == 'H' and orien == '180':
                    image = cv2.rotate(image, cv2.ROTATE_180)
                image = cv2.flip(image,0)
                best_image, image = tight_crop_with_padding(best_image,image,5)
                ratio = tight_bounding_box(best_image)
                if 0.46<=ratio<=55:
                    pass
                else:
                    image = split_image_vertically(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(dstn_folder,t.split("/")[-1]),image)
            except:
                skipper+=1
                print(t.split("/")[-1])
                exce.append(t.split("/")[-1])
    print(f"number of files skipped: {len(exce)}")
    with open(dstn_folder.split("/")[-2]+"_"+dstn_folder.split("/")[-1]+"_exceptions_v2.json",'w') as js:
        json.dump(exce,js,indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)