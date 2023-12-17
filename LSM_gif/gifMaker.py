import glob
from PIL import Image


def make_gif(frame_folder):
    imgs = glob.glob(f"{frame_folder}/*.png")
    l = len(imgs)
    frames = [Image.open(f"./{frame_folder}/{i}.png") for i in range(l)]
    #print(frames)
    frame_one = frames[0]
    frame_one.save("lsmRadar02.gif", format="GIF", append_images=frames,
                  save_all=True, duration=20, loop=0)


if __name__ == "__main__":
    make_gif("./radarGif_01")
