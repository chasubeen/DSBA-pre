from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset, DataLoader
import omegaconf
from typing import List, Tuple, Literal


class IMDBDataset(Dataset):
    def __init__(self, data_config: omegaconf.DictConfig, split: Literal['train', 'valid', 'test']):
        """
        Inputs:
            data_config: omegaconf.DictConfig {
                model_name: str
                max_len: int
                valid_size: float
            }
            split: Literal['train', 'valid', 'test']
        Outputs: None
        """
        self.split = split
        self.data_config = data_config
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.model.model_name)

        ## 데이터셋 로드 및 병합
        dataset = load_dataset(data_config.data.dataset_name)
        full_train_data = dataset['train']
        full_test_data = dataset['test']
        
        # Train + Test 데이터 합치기
        combined_data = concatenate_datasets([full_train_data, full_test_data])

        # Train:Valid:Test = 8:1:1로 나누기
        train_test_split = combined_data.train_test_split(test_size=0.2, seed=42)
        train_data = train_test_split["train"]
        temp_data = train_test_split["test"]

        valid_test_split = temp_data.train_test_split(test_size=0.5, seed=42)
        valid_data = valid_test_split["train"]
        test_data = valid_test_split["test"]

        # 데이터 지정
        self.data = {
            "train": train_data,
            "valid": valid_data,
            "test": test_data
        }[split]
        
        ## Tokenization 적용
        # 토큰화 후 원본 텍스트는 더이상 필요 없기에 삭제
        self.data = self.data.map(self.tokenize_function, 
                                  batched = True, 
                                  remove_columns = ["text"])
        self.data = self.data.to_dict()
        
        print(f">> SPLIT: {self.split} | Total Data Length: {len(self.data['input_ids'])}")

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=self.data_config.data.max_len
        )

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx) -> dict:
        """
        Inputs:
            idx: int
        Outputs:
            input_data: dict {
                input_ids: torch.Tensor
                token_type_ids: torch.Tensor
                attention_mask: torch.Tensor
                label: torch.Tensor
            }
        """
        input_data = {key: torch.tensor(self.data[key][idx], dtype=torch.long) for key in self.data if key != "token_type_ids"}
        
        # BERT 모델일 때만 token_type_ids 포함
        if "token_type_ids" in self.data:
            input_data["token_type_ids"] = torch.tensor(self.data["token_type_ids"][idx], dtype=torch.long)
        
        return input_data

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """
        Inputs:
            batch: List[dict]
        Outputs:
            data_dict: dict {
                input_ids: torch.Tensor
                token_type_ids: torch.Tensor
                attention_mask: torch.Tensor
                label: torch.Tensor
            }
        """
        data_dict = {key: torch.stack([data[key] for data in batch]) for key in batch[0]}
        return data_dict


def get_dataloader(data_config: omegaconf.DictConfig, split: Literal['train', 'valid', 'test']) -> DataLoader:
    """
    Output: torch.utils.data.DataLoader
    """
    dataset = IMDBDataset(data_config, split)
    
    return DataLoader(
        dataset,
        batch_size=data_config.data.batch_size[split],
        shuffle=(split == 'train'),
        collate_fn=IMDBDataset.collate_fn,
        pin_memory=data_config.data.pin_memory
    )