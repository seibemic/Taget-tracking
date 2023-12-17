import glob
from PIL import Image


def make_gif(frame_folder):

    l = 3000#len(imgs)
    frames = [Image.open(f"./{frame_folder}/{i}.png") for i in range(0,l)]
    frame_one=frames[0]
    frame_one.save("lsmRadar01.gif", format="GIF", append_images=frames,
                   save_all=True, duration=20, loop=0)
    for fr in frames:
        fr.close()

    # for j in range(500, 3000, 500): #14400
    #     frames = [Image.open(f"./{frame_folder}/{i}.png") for i in range(j, min(j+500,14399))]
    #     frame_one = Image.open(f"lsmRadar01.gif")
    #     frame_one.save("lsmRadar01.gif", format="GIF", append_images=frames,
    #                    save_all=True, duration=20, loop=0)
    #     frame_one.close()
    #     for fr in frames:
    #         fr.close()

def make_gif2(frame_folder):

    k=12000
    l = 14400#len(imgs)
    frames = [Image.open(f"./{frame_folder}/{i}.png") for i in range(k,l)]
    frame_one=Image.open(f"lsmRadar04.gif")
    frame_one.save("lsmRadar05.gif", format="GIF", append_images=frames,
                   save_all=True, duration=20, loop=0)
    for fr in frames:
        fr.close()

    # for j in range(500, 3000, 500): #14400
    #     frames = [Image.open(f"./{frame_folder}/{i}.png") for i in range(j, min(j+500,14399))]
    #     frame_one = Image.open(f"lsmRadar01.gif")
    #     frame_one.save("lsmRadar01.gif", format="GIF", append_images=frames,
    #                    save_all=True, duration=20, loop=0)
    #     frame_one.close()
    #     for fr in frames:
    #         fr.close()





if __name__ == "__main__":
   make_gif2("./radarGif_01")

