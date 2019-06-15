import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from tqdm import tqdm

def crop_frame(image, obsPos, size, cropSize):
    cropDia = int(cropSize/2)
    x = min(max(obsPos[0], 1), size[1]-2)
    y = min(max(obsPos[1], 1), size[0]-2)
    newImg = image[max(0,y-cropDia):min(y+cropDia, size[0]-1), max(0,x-cropDia):min(x+cropDia, size[1]-1)]
    if x<cropDia:
        newImg = np.append(np.zeros((newImg.shape[0],cropDia-x,3)).astype(int),newImg,1)
    if y<cropDia:
        newImg = np.append(np.zeros((cropDia-y, newImg.shape[1], 3)).astype(int), newImg, 0)
    if (size[0]-y)<cropDia+1:
        newImg = np.append(newImg, np.zeros((cropDia+1-size[0]+y,newImg.shape[1],3)).astype(int), 0)
    if (size[1]-x)<cropDia+1: newImg = np.append(newImg, np.zeros((newImg.shape[0],cropDia+1-size[1]+x,3)).astype(int), 1)
    return newImg

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data-file', help="path to contact data")
    ap.add_argument('-s', '--session-dir', help="path to sessions")
    ap.add_argument('-o', '--output-dir', default='images',
                    help="path to output images (default: images)")
    ap.add_argument('-c', '--crop-size', type=int, default=168,
                    help="size of output image (default: 168), should be even")
    args = vars(ap.parse_args())

    with open(args['data_file']) as f:
        length = len(f.readlines()) - 1

    frames = open(args['data_file'])
    frames.readline()

    clips = {}
    tracked_features = {}
    widths = {}

    for frame_info in tqdm(frames, total=length):
        if frame_info.strip() == '':
            continue

        session, frame_num, class_name = frame_info.split(',')
        frame_num = int(frame_num)
        class_name = class_name.strip()

        if session not in clips:
            clip_loc = os.path.join(args['session_dir'], session, 'runTop.mp4')
            clips[session] = VideoFileClip(clip_loc)
            widths[session] = len(str(
                              clips[session].duration * clips[session].fps))
            print("Loaded session", session)

        if session not in tracked_features:
            tf_loc = os.path.join(args['session_dir'], session,
                                  'trackedFeaturesRaw.csv')
            tracked_features[session] = np.loadtxt(tf_loc, delimiter=',',
                                                   skiprows=1)

        size = clips[session].size[:1][::-1]

        output_path = os.path.join(args['output_dir'], class_name)
        if not os.path.exists(output_path):
            os.makedirs(os.path.join(output_path))

        features = tracked_features[session][frame_num]
        image = clips[session].get_frame(framenum / clips[session].fps)
        obs_pos = [int(features[22]), int(features[23])]
        cropped_image = crop_frame(image, obs_pos, size,
                                   args['crop_size']).astype(np.uint8)

        plt.imsave(os.path.join(output_path,
                                str(frame_num).zfill(widths[session])),
                   cropped_image)
