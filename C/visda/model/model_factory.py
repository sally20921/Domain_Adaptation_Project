from model import resnet

__sets = {}

for model_name in ['resnet50']:
    if model_name in ['resnet50']:
         eval_str = "resnet.{}"
        __sets[model_name] = (
            lambda num_classes, in_features, pretrained, num_domains, model_name=model_name, eval_str=eval_str:
            eval(eval_str.format(model_name))(pretrained=pretrained, num_classes=num_classes,
                                              in_features=in_features))


def get_model(model_name, num_classes, in_features=0, num_domains=2, pretrained=False):
    model_key = model_name
    if model_key not in __sets:
        raise KeyError(
            'Unknown Model: {}, num_classes: {}, in_features: {}'.format(model_key, num_classes, in_features))
    return __sets[model_key](num_classes=num_classes, in_features=in_features,
                             pretrained=pretrained, num_domains=num_domains)

