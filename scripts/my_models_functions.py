import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

import prepare_data_functions as prd

class CNN_Net(nn.Module):
        def __init__(self, image_height=18, image_width=24):
            super().__init__()

            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            h, w = calculate_output_size(image_height, image_width)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * h * w, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 6)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)

            x =  self.flatten(x)

            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)

            return x

def calculate_output_size(image_height: int, image_width: int) -> tuple[int, int]:
    """
    Calculates the output size after two convolution and pooling layers.
    
    Parameters:
    - image_height (int): Height of the input image.
    - image_width (int): Width of the input image.
    
    Returns:
    - tuple[int, int]: Height and width of the image after the layers.
    """
    
    def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
        return (size - kernel_size + 2 * padding) // stride + 1

    def maxpool2d_size_out(size, kernel_size=2, stride=2):
        return (size - kernel_size) // stride + 1

    conv1_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    h = conv2d_size_out(image_height, **conv1_params)
    w = conv2d_size_out(image_width, **conv1_params)
    
    pool1_params = {'kernel_size': 2, 'stride': 2}
    h = maxpool2d_size_out(h, **pool1_params)
    w = maxpool2d_size_out(w, **pool1_params)
    
    conv2_params = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    h = conv2d_size_out(h, **conv2_params)
    w = conv2d_size_out(w, **conv2_params)

    pool2_params = {'kernel_size': 2, 'stride': 2}
    h = maxpool2d_size_out(h, **pool2_params)
    w = maxpool2d_size_out(w, **pool2_params)
    
    return h, w

def load_my_image_pred_model(model_path: str, image_height=18, image_width=24) -> CNN_Net:
    """
    Loads a trained CNN model from a file.
    
    Parameters:
    - model_path (str): Path to the model file.
    - image_height (int): Height of the input images. Default is 18.
    - image_width (int): Width of the input images. Default is 24.
    
    Returns:
    - CNN_Net: The loaded CNN model.
    """

    model = CNN_Net(image_height, image_width)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def move_image(image_pil: Image.Image, n: int = 10) -> Image.Image:
    """
    Shifts image horizontally, moving part of the image from left to right.
    
    Parameters:
    - image_pil (Image.Image): The input PIL image.
    - n (int): Number of parts to divide the image width. Default is 10.
    
    Returns:
    - Image.Image: The shifted PIL image.
    """

    width, height = image_pil.size
    crop_width = width // n

    left_part = image_pil.crop((0, 0, crop_width, height))
    right_part = image_pil.crop((crop_width, 0, width, height))

    new_image_pil = Image.new('RGB', (width, height))
    new_image_pil.paste(right_part, (0, 0))
    new_image_pil.paste(left_part, (width - crop_width, 0))

    return new_image_pil

def predict_image(image: Image.Image, model: torch.nn.Module) -> tuple[int, float]:
    """
    Predicts the class of an input image using a trained model.
    
    Parameters:
    - image (Image.Image): The input PIL image.
    - model (torch.nn.Module): The trained model.
    
    Returns:
    - tuple[int, float]: Predicted class label and its probability.
    """    
    transform = transforms.Compose([
        transforms.Grayscale(),   # Convert to grayscale
        transforms.ToTensor(),  # Convert to tensor
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    probability = probabilities[predicted_class].item()

    return predicted_class, probability

def get_class_name(class_number: int) -> str:
    """
    Maps a class index to its name.
    
    Parameters:
    - class_number (int): The class index.
    
    Returns:
    - str: The name of the class.
    """
    
    class_names = {
    0: "Sector",
    1: "Part of helicoid",
    2: "Disk",
    3: "Helicoid",
    4: "Enneper",
    5: "Complex structure"
    }
        
    return class_names.get(class_number, "Unknown")


def predict_el_class(image_pil: Image.Image, n_shift: int, model: torch.nn.Module) -> tuple[str, float]:
    """
    Predicts the class of an image with shifting.
    
    Parameters:
    - image_pil (Image.Image): The input PIL image.
    - n_shift (int): Number of shifts to perform.
    - model (torch.nn.Module): The trained model.
    
    Returns:
    - tuple[str, float]: Predicted class name and its probability.
    """
    preds = {}
    for i in range(n_shift):
        pred, prob = predict_image(image_pil, model)
        preds[i] = (pred, prob)
        image_pil = move_image(image_pil, n_shift)

    class_weights = {}
    for prediction in preds.values():
        predicted_class, probability = prediction
        class_weights[predicted_class] = class_weights.get(predicted_class, 0) + probability

    element = max(class_weights, key=class_weights.get)
    normalized_probability = class_weights[element] / sum(class_weights.values())
    el_name = get_class_name(int(element))

    return el_name, normalized_probability

def calculate_weighted_vote(dots: np.ndarray, models: dict, model_errors: dict) -> tuple[str, float]:
    """
    Calculates the weighted vote for a set of points using multiple models.
    
    Parameters:
    - dots (np.ndarray): Array of points with their coordinates and types.
    - models (dict): Dictionary of models.
    - model_errors (dict): Dictionary of model errors for different classes.
    
    Returns:
    - tuple[str, float]: Final class name and its probability.
    """
    predictions = []

    for model_name, model in models.items():
        
        if model_name == 'model_phi_theta':
            image_pil = prd.get_phi_theta_image(dots)
            n_shift = 10
            el_name, probability = predict_el_class(image_pil, n_shift, model)
            image_pil.close()
            predictions.append((model_name, el_name, probability))

        elif model_name == 'model_x_y':
            image_pil = prd.get_x_y_image(dots)
            n_shift = 1
            el_name, probability = predict_el_class(image_pil, n_shift, model)
            image_pil.close()
            predictions.append((model_name, el_name, probability))            
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    weighted_votes = {}

    for model_name, el_name, probability in predictions:
        error = model_errors[model_name].get(el_name, 1)
        weight = (1 - error) * probability
        
        if el_name in weighted_votes:
            weighted_votes[el_name] += weight
        else:
            weighted_votes[el_name] = weight

    final_class = max(weighted_votes, key=weighted_votes.get)
    final_probability = weighted_votes[final_class] / sum(weighted_votes.values())
    
    return final_class, final_probability

def custom_sort(arr: list[str]) -> list[str]:
    """
    Sorts a list of class names in a predefined order.
    
    Parameters:
    - arr (list[str]): List of class names.
    
    Returns:
    - list[str]: Sorted list of class names.
    """
    order = {
        "Sector": 0,
        "Part of helicoid": 1,
        "Disk": 2,
        "Helicoid": 3,
        "Enneper": 4,
        "Complex structure": 5
    }
    return sorted(arr, key=lambda x: order.get(x, float('inf')))

def map_values(input_arr: list[str]) -> tuple[str, any, any]:
    """
    Maps a list of class names to a structured classification.
    
    Parameters:
    - input_arr (list[str]): List of class names.
    
    Returns:
    - tuple[str, any, any]: Mapped structure and its features.
    """
    elements = ' '.join(input_arr)
    mapping = {
        "Sector": ("Sector", np.NaN, np.NaN),


        "Sector Sector":                    ("Double helicoid", np.NaN, np.NaN),
        "Sector Part of helicoid":          ("Double helicoid", np.NaN, np.NaN),
        "Part of helicoid Part of helicoid":("Double helicoid", np.NaN, np.NaN),
        
        "Disk":                             ("Disk", np.NaN, np.NaN),
        "Sector Disk":                      ("Disk", "with sector", np.NaN),
        "Sector Sector Disk":               ("Disk", "with two sectors", np.NaN),

        "Disk Disk":                        ("Catenoid", np.NaN, np.NaN),
        "Sector Disk  Disk":                ("Catenoid", "with sector", np.NaN),
        "Sector Sector Disk  Disk":         ("Catenoid", "with two sectors", np.NaN),

        "Disk Disk Disk":                   ("Costa", np.NaN, np.NaN),
        "Sector Disk Disk Disk":            ("Costa", "with sector", np.NaN),

        "Helicoid":                         ("Helicoid", np.NaN, np.NaN),
        "Disk Helicoid":                    ("Helicoid", "with disk", np.NaN),
        "Part of helicoid Disk":            ("Helicoid", "with disk", np.NaN),
        "Sector Disk Helicoid":             ("Helicoid", "with disk", "and sector"),
        
        "Enneper":                          ("Enneper", np.NaN, np.NaN),
        "Sector Enneper":                   ("Enneper", "with sector", np.NaN),
        "Sector Sector Enneper":            ("Enneper", "with two sectors", np.NaN),
    
        "Complex structure":                ("Complex structure", np.NaN, np.NaN),
    }
    
    return mapping.get(elements, (elements, np.NaN, np.NaN))

def pred_structure(structure: np.ndarray, models: dict, model_errors: dict) -> tuple[str, any, any]:
    """
    Predicts all the elements of the structure by their points.
    
    Parameters:
    - structure (np.ndarray): Array of points representing the structure.
    - models (dict): Dictionary of models.
    - model_errors (dict): Dictionary of model errors for different classes.
    
    Returns:
    - tuple[str, any, any]: Predicted structure and its features.
    """

    clustered_dots = prd.select_elements_sort_DBSCAN(structure)

    elements = []
    for el_number, el_dots in clustered_dots:

        dots_B = el_dots[el_dots[:, 3] == 2.]
        pred_label, _ = calculate_weighted_vote(dots_B, models, model_errors)
        elements.append(pred_label)

    sorted_elements = custom_sort(elements)
    structure, feature_1, feature_2 = map_values(sorted_elements)

    return  structure, feature_1, feature_2