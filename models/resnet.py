
import tensorflow.keras as keras

from initializers import PriorProbability
from layers import UpsampleLike
from models.backbone import Backbone
from utils.image import preprocess_image


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """
    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)

    def retinanet(self, num_classes, backbone='resnet50', **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        inputs_base = keras.layers.Input(shape=(None, None, 3))
        input_shape = (None, None, 3)
        resnet = keras.applications.ResNet50(weights="imagenet", include_top=False,input_shape=input_shape,classes=num_classes,input_tensor=inputs_base)
        layer_names = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
        layer_outputs = [resnet.get_layer(name).output for name in layer_names]
        num_anchors = 9
        pyramid_feature_size = 256
        regression_feature_size = 256
        name = 'regression_submodel'
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            'bias_initializer': 'zeros'
        }
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))  # None None 256
        outputs = inputs
        for i in range(4):
            outputs = keras.layers.Conv2D(filters=regression_feature_size, activation='relu',
                                          name='pyramid_regression_{}'.format(i), **options)(outputs)
        outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
        outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)
        default_regression_model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
        #num_classes = 1
        prior_probability = 0.01
        classification_feature_size = 256
        name = 'classification_submodel'
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = keras.layers.Conv2D(filters=classification_feature_size, activation='relu',
                                          name='pyramid_classification_{}'.format(i),
                                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                             seed=None),
                                          bias_initializer='zeros',
                                          **options
                                          )(outputs)

        outputs = keras.layers.Conv2D(filters=num_classes * num_anchors, kernel_initializer=keras.initializers.zeros(),
                                      bias_initializer=PriorProbability(probability=prior_probability),
                                      name='pyramid_classification',
                                      **options
                                      )(outputs)
        outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
        outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
        default_classification_model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
        backbone_layers = layer_outputs
        name = 'retinanet'
        submodels = [('regression', default_regression_model),
                     ('classification', default_classification_model)
                     ]
        C3, C4, C5 = backbone_layers
        feature_size = 256
        # upsample C5 to get P5 from the FPN paper
        P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
        P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
        P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

        # add P5 elementwise to C4
        P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
        P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
        P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
        P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

        # add P4 elementwise to C3
        P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
        P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
        P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
        P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

        features = [P3, P4, P5, P6, P7]
        pyramids = []
        for n, m in submodels:
            list_models = []
            for f in features:
                list_models.append(m(f))
            pyramids.append(keras.layers.Concatenate(axis=1, name=n)(list_models))
        backbone_retinanet = keras.models.Model(inputs=inputs_base, outputs=pyramids, name='retinanet')
        return backbone_retinanet

        #model = backbone_retinanet
        #training_model = model
        #coding of retinanet_bbbox
        #inputs_base -->inputs

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')




