"""Inference-only checkpoint loading for the existing ordered dense Flow.

No imports from training scripts or exact geometry checker. Sampler/decoder
are the same OrderedDenseFieldFlow deployed throughout the project.
"""
import inspect
import torch
from p37_model import OrderedDenseFieldFlow


def load_flow_prior(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=state.get('config',{})
    names=inspect.signature(OrderedDenseFieldFlow.__init__).parameters
    kwargs={key:value for key,value in config.items() if key in names and key not in ('self','kwargs')}
    model=OrderedDenseFieldFlow(**kwargs).to(device)
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model,state
