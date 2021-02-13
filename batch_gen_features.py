from feature_processor import FeatureProcessor

data_prepath = './data_prcsd/'
output_prepath = './ftrs_prcsd/'

extractor_cls = FeatureProcessor(data_dir=data_prepath, output_dir=output_prepath, acc_only=True, is_eval=False)
##Extract features and labels based on the predifined window size and hop length
extractor_cls._featlabel_extractor()
##Nomralize the extratced spectrogram
extractor_cls._features_processor()