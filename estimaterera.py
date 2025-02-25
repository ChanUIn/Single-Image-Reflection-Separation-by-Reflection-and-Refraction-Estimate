import os
import cv2
from tqdm import tqdm
from Get_Map import get_map


GT_dir = r'./t/'  #GT
mix_dir = r'./I/' # input
output_dir = r'./dir'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'estimate.txt')

with open(output_file, 'w') as f:
    count = 13701  # start number
    gt_filelist = sorted([file for file in os.listdir(GT_dir) if file.lower().endswith('.jpg')])
    mix_filelist = sorted([file for file in os.listdir(mix_dir) if file.lower().endswith('.jpg')])

    for gt_filename, mix_filename in tqdm(zip(gt_filelist, mix_filelist), total=len(gt_filelist), desc="Processing images"):
        gt_image = cv2.imread(os.path.join(GT_dir, gt_filename))
        mix_image = cv2.imread(os.path.join(mix_dir, mix_filename))

        if gt_image is not None and mix_image is not None:
            map_A, map_B = get_map(gt_image.shape[0], gt_image.shape[1])
            B = map_B.mean()

            f.write(f'{count:05d}.jpg\t{map_A.mean()}\t{B}\n')
            count += 1

print(f'File {output_file} generated successfully.')
