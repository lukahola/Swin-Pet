import torch
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
from optimizer import build_optimizer
from models import swin_transformer
from models import build_model
from utils import load_checkpoint, load_pretrained, save_checkpoint

config_file = '../configs/swin_base_batch4_window7_224.yaml'
ckpt_path = r'D:/Downloads/ckpt_epoch_260_39.pth'
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
classes = ("0Abyssinian","1American_shorthair","2Bengal","3Birman","4Bombay","5British_Shorthair","6Egyptian_Mau",
           "7Maine_Coon","8Persian","9Poddle","10Ragdoll","11Russian_Blue","12Siamese","13Sphynx","14american_bulldog",
           "15american_pit_bull_terrier","16basset_hound","17beagle","18boxer","19chihuahua","20english_cocker_spaniel",
           "21english_setter","22german_shorthaired","23great_pyrenees","24havanese","25japanese_chin","26keeshond",
           "27leonberger","28miniature_pinscher","29newfoundland","30pomeranian","31pug","32saint_bernard","33samoyed",
           "34scottish_terrier","35shiba_inu","36staffordshire_bull_terrier","37wheaten_terrier","38yorkshire_terrier"
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('D:/Downloads/ckpt_epoch_260_39.pth', map_location='cpu')
config = checkpoint['config']
model = build_model(config)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(DEVICE)

# path = 'G:/ln/imagenet/val/Ragdoll/'
# testList = os.listdir(path)
# for file in testList:
#     img = Image.open(path + file)
#     # for iamges in os.listdir(path + file + '/'):
#     #     img = Image.open(path + file + '/' + iamges)
#     #     cvimg = cv2.imread(path + file, 1)
#         # cv2.imshow(file, cvimg)
#         # cv2.waitKey(0)
#         # cv2.destroyWindow(file)
#     img = transform_test(img)
#     img.unsqueeze_(0)
#     img = Variable(img).to(DEVICE)
#     out = model(img)
#     # Predict
#     _, pred = torch.max(out.data, 1)
#     print('Image Name:{},id:{},predict:{}'.format(file, pred.data.item(), classes[pred.data.item()]))

path = './demo_image/'
img_name = 'arr.jpg'
o_img = Image.open(path + img_name)
img = transform_test(o_img)
img.unsqueeze_(0)
img = Variable(img).to(DEVICE)
out = model(img)
_, pred = torch.max(out.data, 1)

print('Image Name:{},predict:{}'.format(img_name, classes[pred.data.item()]))
class_name = classes[pred.data.item()]

cvimg = cv2.imread(path + img_name, 1)
cv2.namedWindow(class_name, cv2.WINDOW_NORMAL)
cv2.imshow(class_name, cvimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

