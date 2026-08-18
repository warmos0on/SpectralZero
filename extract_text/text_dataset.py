def get_dataset(dataset_name):
    if dataset_name == "Indian":
        label_values = {
            "seen": {
                "Grass-trees": 6,
                "Buildings-Grass-Trees-Drives": 15,
                "Soybean-notill": 10,
                "Corn-notill": 2,
                "Grass-pasture-mowed": 7,
                "Corn-mintill": 3,
                "Corn": 4,
                "Woods": 14,
                "Soybean-mintill": 11,
                "Soybean-clean": 12,
                "Stone-Steel-Towers": 16,
                "Hay-windrowed": 8,
                "Wheat": 13,
            },
            "unseen": {
                "Alfalfa": 1,
                "Oats": 9,

                "Grass-pasture": 5,

            }
        }
    elif dataset_name == "LongKou":
        label_values = {
            "seen": {
                "Corn": 1,
                "Cotton": 2,
                "Sesame": 3,
                "Broad-leaf soybean": 4,
                "Narrow-leaf soybean": 5,
                "Rice": 6,
                "Water": 7,
            },
            "unseen": {
                "Mixed weed": 9,
                "Roads and houses": 8
            }
        }
    elif dataset_name == "Houston2018":
        label_values = {
            "seen": {
                "Healthy grass": 1,
                "Stressed grass": 2,
                "Evergreen trees": 4,
                "Deciduous trees": 5,
                "Bare earth": 6,
                "Residential buildings": 8,
                "Non-residential buildings": 9,
                "Roads": 10,
                "Sidewalks": 11,
                "Crosswalks": 12,
                "Major thoroughfares": 13,
                "Highways": 14,
                "Railways": 15,
                "Paved parking lots": 16,
                "Cars": 18,
                "Trains": 19,
                "Unpaved parking lots": 17,

            },
            "unseen": {
                "Artificial turf": 3,
                "Water": 7,
                "Stadium seats": 20,

            }
        }
    elif dataset_name == "HanChuan":
        label_values = {
            "seen": {
                "Corn": 1,
                "Cotton": 2,
                "Narrow-leaf soybean": 5,
                "Rice": 6,
                "Water": 7,
                "Roads and houses": 8,
                "Mixed weed": 9,
            },
            "unseen": {
                "Broad-leaf soybean": 4,
                "Sesame": 3,
            }
        }
    elif dataset_name == "Houston":
        label_values = {
            "seen": {
                "Stressed grass": 2,  # t-SHE ×
                "Railways": 11,        # t-SHE ×
                "Highways": 10,        # t-SHE ×
                "Road": 9,             # t-SHE ×
                "Commercial": 8,       # t-SHE ×
                "Parking Lot 1": 12,   # t-SHE ×
                "Parking Lot 2": 13,   # t-SHE ×
                "Healthy grass": 1,    # t-SHE ×
                "Trees": 4,            # t-SHE ×
                "Soil": 5,  # t-SHE ×
                "Residential": 7,  # t-SHE ×√
                "Running Track": 15,  # t-SHE √

            },
            "unseen": {
                "Water": 6,  # t-SHE √
                "Tennis Court": 14,  # t-SHE √
                "Synthetic grass": 3,  # t-SHE √

            }
        }
    elif dataset_name == "PaviaU":
        label_values = {
            "seen": {
                "Asphalt": 1,
                "Self-Blocking Bricks": 8,
                "Shadows": 9,
                "Bitumen": 7,
                "Meadows": 2,
                "Painted metal sheets": 5,

            },
            "unseen": {
                "Trees": 4,
                "Bare Soil": 6,
                "Gravel": 3,

            }
        }
    else:
        ValueError("Wrong dataset name!")
    class_num = len(label_values["seen"])
    return class_num, label_values