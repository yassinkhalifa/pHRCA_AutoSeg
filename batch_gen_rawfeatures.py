from feature_processor import RawFeatureProcessor

data_prepath = './data_prcsd/'
output_prepath = './rawftrs_prcsd/'

extractor_cls = RawFeatureProcessor(data_dir=data_prepath, output_dir=output_prepath, acc_only=True, is_eval=False)
##Extract features and labels based on the predifined window size and hop length
extractor_cls._featlabel_extractor()