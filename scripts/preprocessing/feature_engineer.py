from .canny_edge_generator import CannyEdgeGenerator
from .canny_point_extractor import CannyFeatureExtractor

class FeatureEngineer():

    _DEFAULT_CONFIGS = {
            "generator": {
                'in_range': {'lower_bounds': 150, 'upper_bounds': 255},
                'canny': {'weak_edge': 50, 'sure_edge': 100, 'blur_ksize': 3, "blur_order": "after"},
            },
            'extractor': {"filter_type": "median", "n_std": 2.0, "weight": 5}
        }
    
    def __init__(self, x_mid, configs:dict=None):
        if configs is None:
            gen_configs, ext_configs = self._DEFAULT_CONFIGS["generator"], self._DEFAULT_CONFIGS["extractor"]
        
        else:
            gen_configs, ext_configs = configs["generator"], configs["extractor"]

            self.generate = CannyEdgeGenerator(gen_configs)
            self.extract = CannyFeatureExtractor(x_mid, ext_configs)

    def generate_features(self, frame):
        thresh, edge_map = self.generate.generate(frame)
        kps = self.extract.extract(edge_map)
        return thresh, edge_map, kps