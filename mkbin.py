from PIL import Image
from glob import glob
import pickle

acc = 1

def class2num(st):
    if "backward" in st:
        return 0
    elif "forward" in st:
        return 1
    elif "left" in st:
        return 2
    elif "right" in st:
        return 3
    elif "stop" in st:
        return 4
    return 5
        
anno = {}

for f in glob("*/*.jpg"):
    filename = f.split("/")[-1]
    out_file_name = '{:05d}'.format(acc) + ".png"
    Image.open(f).resize((320,320)).save(out_file_name, "PNG")
    anno[acc] = class2num(f)
    print(acc, anno[acc])
    acc += 1 

with open("anno.pkl", "wb") as ap:
    pickle.dump(anno, ap)
