import pickle


# Classes
class Bidict(dict):
    """
    A bidirectional dictionary object. It creates manages two dictionaries:
        1. the first one has a normal "key to value" mapping
        2. the second one is an inversed verions "value to key" pairs.
    """
    def __init__(self, *args, **kwargs):
        """
        We inherit from the built-in dictionary class
        :param args: typical dictionary arguments
        :param kwargs: typical dict keyword arguments
        """
        super(Bidict, self).__init__(*args, **kwargs)
        self.inverse = {}  # create an inverse dict
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        """
        Each time a pair is set up, this class adds the reverse referenced pair to the inverse dict
        :param key: a key for dict pair
        :param value: the value associated with the given key
        :return: None
        """
        if key in self:
            self.inverse[self[key]].remove(key)
        super(Bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        """
        Each time we remove an element from the main dict, we also remove its reverse pair from the inverse dict
        :param key: the key to be removed (its value is deleted also)
        :return: None
        """
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(Bidict, self).__delitem__(key)


# Constants
# This reference Bidict maps the directory names to corresponding model names. This can be automated but for the current
# purposes it will suffice.
ref_dict = Bidict(
    {
        'ensemble_v3_TF_TranspFINETUNED_TranspTF_pred_imgs': 'U-NET Ensemble 1',
        'ensemble_v3_v2_TranspFINETUNED_pred_imgs': 'U-NET Ensemble 2',
        'MobNet_um_kitti_AUG_pred_imgs_combined': 'MobileNet AUG',
        'MobNet_um_kitti_simple_pred_imgs_combined': 'MobileNet simple',
        'UNET_um_kitti_AUG_deconv0.5_pred_imgs_combined': 'U-NET Transposed G-0.5',
        'UNET_um_kitti_AUG_deconv_pred_imgs_combined': 'U-NET Transposed G-1',
        'UNET_um_kitti_AUG_pred_imgs_combined': 'U-NET AUG',
        'UNET_um_kitti_BCE_pred_imgs_combined': 'U-NET BCE',
        'UNET_um_kitti_simple_pred_imgs_combined': 'U-NET simple',
        'UNET_um_umm_kitti_AUG_DECONV1_pred_imgs_combined': 'U-NET Transposed TF',
        'UNET_um_umm_kitti_AUG_DECONV1_TF_FINE_GR_pred_imgs_combined': 'U-NET Transposed FINETUNED',
        'UNET_um_umm_kitti_AUG_v1_pred_imgs_combined': 'U-NET TF v1',
        'UNET_um_umm_kitti_AUG_v2_pred_imgs_combined': 'U-NET TF v2',
        'UNET_um_umm_kitti_AUG_v3_pred_imgs_combined': 'U-NET TF v3',
        'test predictions_pred_imgs': 'Test Model'
    }
)


# Helpers
def pretty(d, indent=0, title=''):
    """
    A method for printing dictionary hierarchy.
    :param d: a dictionary object to iterate
    :param indent: how many tabs to use to indent each level of data
    :param title: a title for the printed dictionary
    :return: None
    """
    if title:
        print(title)
        print('_' * 100)
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def load_pickled_dict(pickled_dict_path):
    """
    This method loads a dictionary object saved as a pickle and returns it
    :param pickled_dict_path: a path to the saved pickled dictionary object
    :return: A dictionary
    """
    return pickle.load(open(pickled_dict_path, 'rb'))
