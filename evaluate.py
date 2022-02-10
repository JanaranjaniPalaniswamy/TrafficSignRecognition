import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from model import TrafficSignNet
from data import get_test_loader
from torchvision.utils import make_grid
from train import valid_batch
from tqdm import tqdm


def evaluate(model, loss_func, dl):
    """
    The function to evaluate the trained model with test set and to calculate test accuracy and loss
    :param model: Trained model checkpoint file
    :param loss_func: Cross entropy loss
    :param dl: Test dataset
    :return: Function return after calculating the total Test loss and accuracy
    """
    model.eval()
    with torch.no_grad():
        # Calculation of loss, correct predictions and number of predictions
        losses, corrects, nums = zip(
            *[valid_batch(model, loss_func, x, y) for x, y in tqdm(dl)])
        # Test loss and accuracy calculation
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

        print(f"Test loss: {test_loss:.6f}\t"
              f"Test accruacy: {test_accuracy:.3f}%")

# TODO: Comment-out this function when visualize_stn(dl, outfile) function is disabled

def convert_image_np(img):
    """
    The function converts grid of test data image or transformed STN layer output image to  numpy format
    :param img: Grid of input image tensors
    :return: Numpy data
    """
    img = img.numpy().transpose((1, 2, 0)).squeeze()
    return img


# TODO: Comment-out this function when STN layer is disabled in the model

def visualize_stn(dl, outfile):
    """
    Function to visualize the output of the STN layer in a grid format
    :param dl: Test dataset
    :param outfile: File to store the output image
    :return: Returns after saving the image plot to a file
    """
    with torch.no_grad():
        data = next(iter(dl))[0]

        input_tensor = data.cpu()
        transformed_tensor = model.stn(data).cpu()

        input_grid = convert_image_np(make_grid(input_tensor))
        transformed_grid = convert_image_np(make_grid(transformed_tensor))

        # Plot the results side-by-side
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((16, 16))
        ax[0].imshow(input_grid)
        ax[0].set_title('Dataset Images')
        ax[0].axis('off')

        ax[1].imshow(transformed_grid)
        ax[1].set_title('Transformed Images')
        ax[1].axis('off')
        # Saves the output image grids in a file
        plt.savefig(outfile)


if __name__ == "__main__":
    # Evaluation settings
    parser = argparse.ArgumentParser(
        description='Traffic sign recognition evaluation script')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located. test.p need to be found in the folder (default: data)")
    parser.add_argument('--model', type=str, default='model.pt', metavar='M',
                        help="the model file to be evaluated. (default: model.pt)")
    parser.add_argument('--outfile', type=str, default='visualize_stn.png', metavar='O',
                        help="visualize the STN transformation on some input batch (default: visualize_stn.png)")

    args = parser.parse_args()

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    # Neural Network and Loss Function
    model = TrafficSignNet().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Data Initialization and Loading
    test_loader = get_test_loader(args.data, device)
    # Testing
    evaluate(model, criterion, test_loader)
    # Data Visualization
    # TODO: Comment the below function call when STN layer is removed in model
    visualize_stn(test_loader, args.outfile)
