from model_func import *
from utility import *

args = define_predict_args()

# Loading the checkpoint
model = load_checkpoint(args)
imagePath = args["image_path"]

# # Inference for classification
#imagePath = "flowers/test/1/image_06754.jpg"
cat_to_name = get_category_name(args["category_names"])
if args["gpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
probs, classes = predict(imagePath, model, args["top_k"], device)
classNames = class_num_to_name(classes, cat_to_name)

for i in range(len(classNames)):
    print(f"{(i+1)}- Class name: {classNames[i]}, Probability: {(probs[i]*100):.2f}% ")

