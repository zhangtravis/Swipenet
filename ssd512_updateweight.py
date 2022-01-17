from keras import backend as K
from keras.optimizers import Adam
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator_train import Evaluator

img_height = 512
img_width = 512
n_classes = 3
model_mode = 'inference'
modelname='ssd512_2013_adam16_0.0001_time3_epoch-01_loss-84.2419_val_loss-11.6939.h5'
K.clear_session()
model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.3, 0.15, 0.07, 0.04],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0]],
                two_boxes_for_ar1=True,
                steps=[16, 8, 4],
                offsets=[0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

weights_path = '/data/deeplearn/SWEIPENet/' + modelname
model.load_weights(weights_path, by_name=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

dataset = DataGenerator()
Pascal_VOC_dataset_images_dir = '/data/deeplearn/SWEIPENet/dataset/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = '/data/deeplearn/SWEIPENet/dataset/Annotations/'
Pascal_VOC_dataset_image_set_filename = '/data/deeplearn/SWEIPENet/dataset/ImageSets/Main/trainval.txt'
classes = ['background', 'seacucumber', 'seaurchin', 'scallop']
dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)
results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=32,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)
