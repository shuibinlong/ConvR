import torch
import logging
from .ConvE import ConvE
from .ConvR import ConvR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_dict = {
    'ConvE': ConvE,
    'ConvR': ConvR
}

def init_model(config):
    # initialize model
    device = config.get('device')
    if config.get('model_name') in model_dict:
        model = model_dict[config.get('model_name')].init_model(config)
    else:
        raise ValueError('Model not support: ' + config.get('model_name'))
    logging.info(model)

    '''
    # For simplicity, use DataParallel wrapper to use multiple GPUs.
    if device == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f'{torch.cuda.device_count()} GPUs are available. Let\'s use them.')
        model = torch.nn.DataParallel(model)
    '''

    model = model.to(device)
    logging.info(f'model loaded on {device}')
    
    return model, device
