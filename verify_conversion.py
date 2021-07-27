import argparse
import json
import sentencepiece as spm

import torch

from fnet import FNetForPreTraining


def compare_output(jax_checkpoint_path, torch_statedict_path, torch_config_path, vocab_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)
    tokenizer.SetEncodeExtraOptions("")

    with open(torch_config_path) as f:
        fnet_torch_config = json.load(f)
    fnet_torch = FNetForPreTraining(fnet_torch_config)
    statedict = torch.load(torch_statedict_path, map_location=torch.device('cpu'))
    fnet_torch.load_state_dict(statedict)
    fnet_torch.eval()

    input_ids, token_type_ids, mlm_positions = get_input(tokenizer, fnet_torch_config['max_position_embeddings'])
    fnet_torch_output = fnet_torch(input_ids, token_type_ids, mlm_positions)


def get_input(tokenizer, seq_len):
    text = "Joseph Harold Greenberg (May 28, 1915 – May 7, 2001) was an American linguist, " \
           "known mainly for his work concerning " \
           "linguistic typology and the genetic classification of languages."

    cls_id = tokenizer.PieceToId("[CLS]")
    mask_id = tokenizer.PieceToId("[MASK]")
    sep_id = tokenizer.PieceToId("[SEP]")
    pad_id = tokenizer.pad_id()

    token_ids = [cls_id] + tokenizer.EncodeAsIds(text) + [sep_id]
    input_ids = torch.full((1, seq_len), pad_id, dtype=torch.long)
    input_ids[0, :len(token_ids)] = torch.LongTensor(token_ids)

    # mask some tokens
    mlm_positions = torch.LongTensor([1, 5, 7])
    input_ids[0, mlm_positions] = mask_id

    token_type_ids = torch.full((1, seq_len), 0, dtype=torch.long)

    max_mlm_maskings = 80
    full_mlm_positions = torch.full((1, max_mlm_maskings), 0, dtype=torch.long)
    full_mlm_positions[:, :len(mlm_positions)] = mlm_positions

    return input_ids, token_type_ids, full_mlm_positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--jax', type=str, required=True, help='path to FNet jax checkpoint')
    parser.add_argument('--torch', type=str, required=True, help='path to PyTorch statedict checkpoint')
    parser.add_argument('--config', type=str, required=True, help='path to PyTorch checkpoint config')
    parser.add_argument('--vocab', type=str, required=True, help='path to vocab file')

    args = parser.parse_args()

    compare_output(args.jax, args.torch, args.config, args.vocab)
