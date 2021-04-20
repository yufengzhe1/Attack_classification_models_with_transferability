from torchvision import models


# load models from torchvision.models, you also can load your own models
def load_models(source_model_names, device):
    source_models = []
    for model_name in source_model_names:
        print("Loading: {}".format(model_name))
        source_model = models.__dict__[model_name](pretrained=True).eval()
        for param in source_model.parameters():
            param.requires_grad = False
        source_model.to(device)
        source_models.append(source_model)
    return source_models


# calculate the ensemble logits of models
def get_logits(X_adv, source_models, norm):
    logits = 0
    for source_model in source_models:
        logits += source_model(norm(X_adv))  # ensemble

    logits /= len(source_models)
    return logits




