from model_func import *
from utility import *

args = define_train_args()

# Load the data
train_data, valid_data, test_data = load_data(args["data_dir"])

# Using the image datasets and the trainforms, define the dataloaders
trainloader = get_data_loader(train_data, True)
validloader = get_data_loader(valid_data, False)
testloader = get_data_loader(test_data, False)

cat_to_name = get_category_name('cat_to_name.json')

epochs = args["epochs"]

# Get model instance
if args["gpu"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model, optimizer, criterion = create_model(cat_to_name, device = device, args = args)
model = train_model(model, optimizer, criterion, trainloader, validloader, device = device, epochs=epochs, print_every=20)

# Save the checkpoint
save_checkpoint('checkpoint.pth', train_data.class_to_idx, epochs, optimizer.state_dict(), model.state_dict(), args)

# Do validation on the test set
test_model(model, criterion, testloader, device)
