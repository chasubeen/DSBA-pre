from transformers import AutoModel
import torch
import torch.nn as nn

from typing import Tuple
import omegaconf


## BERT
class BERTForClassification(nn.Module):
    def __init__(self, model_config: omegaconf.DictConfig):
        super(BERTForClassification, self).__init__()

        # configs
        self.model_name = model_config.model.model_name
        self.num_labels = model_config.model.num_labels
        self.hidden_size = model_config.model.hidden_size
        self.dropout_rate = model_config.model.dropout_rate
        
        # 모델 불러오기
        self.model = AutoModel.from_pretrained(self.model_name)
        # Dropout 레이어
        self.dropout = nn.Dropout(self.dropout_rate)
        # Classification Head
        self.classifier = nn.Linear(self.hidden_size, self.num_labels) # (768, 2)
        
        # loss fn
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BERT 모델 → token_type_ids 필요
        """
        outputs = self.model(input_ids = input_ids, 
                             attention_mask = attention_mask, 
                             token_type_ids = token_type_ids)
        
        # 최종 representation 추출
        pooled_output = torch.mean(outputs.last_hidden_state, dim = 1) # 수정
        
        # Dropout을 적용한 후 classification head에 입력하여 logit값 반환
        logits = self.classifier(self.dropout(pooled_output))
        
        # loss 계산
        loss = self.loss_fn(logits, label)

        return logits, loss


## modernBERT
class ModernBERTForClassification(nn.Module):
    def __init__(self, model_config: omegaconf.DictConfig):
        super(ModernBERTForClassification, self).__init__()

        # configs
        self.model_name = model_config.model.model_name
        self.num_labels = model_config.model.num_labels
        self.hidden_size = model_config.model.hidden_size
        self.dropout_rate = model_config.model.dropout_rate

        # 모델 불러오기
        self.model = AutoModel.from_pretrained(self.model_name)
        # Dropout 레이어
        self.dropout = nn.Dropout(self.dropout_rate)
        # Classification Head
        self.classifier = nn.Linear(self.hidden_size, self.num_labels) # (768, 2)
        
        # loss fn
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ModernBERT 모델 → token_type_ids 없음
        """
        outputs = self.model(input_ids = input_ids, 
                             attention_mask = attention_mask)

        # 최종 representation 추출([CLS] 토큰 활용)
        pooled_output = torch.mean(outputs.last_hidden_state, dim = 1) # 수정
        
        # Dropout을 적용한 후 classification head에 입력하여 logit값 반환
        logits = self.classifier(self.dropout(pooled_output))
        
        # loss 계산
        loss = self.loss_fn(logits, label)

        return logits, loss