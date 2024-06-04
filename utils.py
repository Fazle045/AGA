from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
from torchvision import datasets, models, transforms

###### utils########
def get_input_image_data(folder_name="/home/fa077317/rahat/IMAGENET12/IMAGENET12/Imagenet_20_new/train/", all_classes=None):
    all_classes  = list(all_classes)
    input_dataset = datasets.ImageFolder(folder_name)
    
    
    
    # Convert modified_categories values to tuples
    modified_categories = [(category,) for category in all_classes]
    
    wnid_to_classes = input_dataset.classes
    
    
    # Create wnid_to_classes dictionary
    wnid_to_classes = dict(zip(wnid_to_classes, modified_categories))
    
    input_dataset.wnids = input_dataset.classes
    input_dataset.wnid_to_idx = input_dataset.class_to_idx
    input_dataset.classes = [wnid_to_classes[wnid] for wnid in input_dataset.wnids]
    input_dataset.class_to_idx = {cls: idx for idx, clss in enumerate(input_dataset.classes) for cls in clss}
    
    all_class = tuple(input_dataset.classes[i][0] for i in range(len(input_dataset.classes)))

    return input_dataset, all_class



from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)