from pathlib import Path
from pyfaidx import Fasta
import torch
import os
import time
from hydra.utils import get_original_cwd

class MRNADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        fasta_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        split_ratio=(0.8, 0.1, 0.1),
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval         
        self.split = split

        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval 
        
        # fasta_file = "/raid_elmo/home/lr/zym/data/rna_data/rnacentral_active.fasta"
        fasta_file = Path(fasta_file)
        if not fasta_file.is_absolute():
            fasta_file = Path(get_original_cwd()) / fasta_file
        self.fasta_file = fasta_file
        
        assert sum(split_ratio) == 1, "Split ratios must sum up to 1"
        assert fasta_file.exists(), 'path to fasta file must exist'

        
        start_time = time.time()
        fasta_path_str = str(self.fasta_file)  # 将Path对象转换为字符串
        # 检查.fai索引文件是否存在
        if not os.path.exists(fasta_path_str + '.fai'):
            # 如果不存在，使用pyfaidx读取fasta文件并创建索引
            print(".fai not exists")
            self.seqs = Fasta(fasta_path_str)
        else:
            # 如果索引文件存在，直接使用它
            print(".fai exists")
            self.seqs = Fasta(fasta_path_str, build_index=False)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"读取fasta文件运行时间:{elapsed_time}秒")

        # save index of the sequence
        self.keys = list(self.seqs.keys())

        num_sequences = len(self.keys)
        train_end = int(num_sequences * split_ratio[0])
        valid_end = train_end + int(num_sequences * split_ratio[1])

        if split == "train":
            self.split_keys = self.keys[:train_end]
        elif split == "valid":
            self.split_keys = self.keys[train_end:valid_end]
        elif split == "test":
            self.split_keys = self.keys[valid_end:]
        else:
            raise ValueError("Invalid split. Expected one of: 'train', 'valid', 'test'")


    def __len__(self):
        return len(self.split_keys)
    
    def _getseq(self, seq):
        rna_length = len(seq)
        start, end = 0, rna_length - 1
        left_padding = right_padding = 0

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, rna_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        if rna_length < self.max_length:
            extra_seq = self.max_length - rna_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > rna_length:
            right_padding = end - rna_length
            end = rna_length
        
        if rna_length > self.max_length:
            end = start + self.max_length

        seq = str(seq[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq
    

    def __getitem__(self, idx):
        # 随机选择一个序列键值
        key = self.split_keys[idx % len(self.split_keys)]
        # seq = str(self.seqs[key][:].seq).replace('T', 'U')
        # seq = str(self.seqs[key][:].seq)
        # print(f"key={key}")
        seq = str(self.seqs[key][:].seq)
        # print(f"seq = {seq}")
        # preprocess sequence 
        # seq = self._getseq(seq)
        # print(seq)
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target