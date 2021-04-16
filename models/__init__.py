def backbone(backbone_name):
    """ Returns a backbone object for the given backbone.
    """
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)
