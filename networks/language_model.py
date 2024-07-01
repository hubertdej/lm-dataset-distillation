from typing import Optional

import torch

from .utils import ReparamModule


class LanguageModelWrapper(ReparamModule):
    def __init__(self, model, state):
        super(LanguageModelWrapper, self).__init__()
        self.model = model
        self.state = state
    
    def forward(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
    
    def forward_with_param(self, new_weights: torch.Tensor, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        with self.unflatten_weight(new_weights):
            return self.forward(input_ids=input_ids, inputs_embeds=inputs_embeds)
    
    def extract_embedding_matrix(self) -> torch.Tensor:
        with self.unflatten_weight(self.flat_w):
            return self.model.transformer.wte.weight.data.clone().detach()