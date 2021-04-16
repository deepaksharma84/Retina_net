import layers
import losses
from callbacks import RedirectModel,Evaluate
from utils.anchors import AnchorParameters
from utils.transform import random_transform_generator
import models
import os
import pandas as pd
import tensorflow.keras as keras
import time
from preprocessing.csv_generator import CSVGenerator


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def main():
    backbone = models.backbone('resnet50')
    # create the generators
    #train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
    random_transform = True
    val_annotations = './data/processed/val.csv'
    annotations = './data/processed/train.csv'
    classes = './data/processed/classes.csv'
    common_args = {
        'batch_size': 8,
        'image_min_side': 224,
        'image_max_side': 1333,
        'preprocess_image': backbone.preprocess_image,
    }
    # create random transform generator for augmenting training data
    if random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.05,
            max_rotation=0.05,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            #min_shear=-0.1,
            #max_shear=0.1,
            min_scaling=(0.8, 0.8),
            max_scaling=(1.2, 1.2),
            flip_x_chance=0.5,
            #flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
    train_generator = CSVGenerator(annotations,classes,transform_generator=transform_generator,**common_args)

    if val_annotations:
        validation_generator = CSVGenerator(val_annotations,classes,**common_args)
    else:
        validation_generator = None

    #train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
    num_classes = 1  # change
    model = backbone.retinanet(num_classes, backbone='resnet50')
    training_model = model

    # prediction_model = retinanet_bbox(model=model)
    nms = True
    class_specific_filter = True
    name = 'retinanet-bbox'
    anchor_params = AnchorParameters.default
    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    anchor = [
        layers.Anchors(
            size=anchor_params.sizes[i],
            stride=anchor_params.strides[i],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]
    anchors = keras.layers.Concatenate(axis=1, name='anchors')(anchor)
    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]  # check
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification] + other)

    outputs = detections

    # construct the model
    prediction_model = keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)

    # end of prediction_model = retinanet_bbox(model=model)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=.0001, nesterov=True, clipnorm=1)
        # , clipnorm=0.001)
    )
    print(model.summary())
    # start of create_callbacks
    #callbacks = create_callbacks(model,training_model,prediction_model,validation_generator,args,)
    callbacks = []
    tensorboard_callback = None
    tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = '',
            histogram_freq         = 0,
            batch_size             = 8,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
    callbacks.append(tensorboard_callback)
    evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback,
                          weighted_average=False)
    evaluation = RedirectModel(evaluation, prediction_model)
    callbacks.append(evaluation)
    makedirs('./snapshots/')
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join('./snapshots/',
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone='resnet50', dataset_type='csv'))
                                                 ,verbose=1,save_best_only=False,monitor="mAP",mode='max')

    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.9,
        patience = 4,
        verbose  = 1,
        mode     = 'auto',
        min_delta  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))
    steps = 2500
    epochs=25

    # start training
    history = training_model.fit(
        generator=train_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    timestr = time.strftime("%Y-%m-%d-%H%M")

    history_path = os.path.join(
        './snapshots/',
        '{timestr}_{backbone}.csv'.format(timestr=timestr, backbone='resnet50', dataset_type='csv')
    )
    pd.DataFrame(history.history).to_csv(history_path)


if __name__ == '__main__':
    main()
