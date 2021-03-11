import os
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from bert.modeling import BertForSequenceClassification
from data_loader import distill_dataloader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def do_eval(model, eval_dataloader, device):
    scores = []
    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids = batch_
            logits = model(input_ids, segment_ids, input_mask)
            probs = F.softmax(logits, dim=1)[:, 1]
            scores.append(probs.detach().cpu().numpy())
    scores = np.concatenate(scores)
    return scores


def save_scores(args, scores):

    query_psgids_map = []
    with open(args.features_file, mode='r') as ref_file:
        for i, line in enumerate(ref_file):
            if i == 0:
                continue
            query_psgids_map.append(line.strip().split(",")[0].split('#'))

    assert len(scores) == len(query_psgids_map)

    with open(args.output_scores_file, 'w') as output_scores_file:
        for idx, score in enumerate(scores):
            query_id, psg_id = query_psgids_map[idx]
            out_str = "{0}\t{1}\t{2}\n".format(query_id, psg_id, score)
            output_scores_file.write(out_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        required=True,
                        help="The device you will run on.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The model for inference.")
    parser.add_argument("--features_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The file contains the features of query-passage pairs."
                             "Format: example_id,input_ids,input_mask,segment_ids,label\n")
    parser.add_argument("--output_scores_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output prediction from BERT ranker."
                             "Format: query_id\tpassage_id\tscore\n")
    parser.add_argument("--cache_file_dir",
                        default='./cache',
                        type=str,
                        required=True,
                        help="The directory where cache the features.")
    parser.add_argument("--eval_batch_size",
                        default=100,
                        type=int,
                        help="Batch size for eval.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    if not os.path.exists(args.cache_file_dir):
        os.makedirs(args.cache_file_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=2)
    model.to(device)

    _, dataloader = distill_dataloader(args, SequentialSampler, args.eval_batch_size)

    scores = do_eval(model, dataloader, device)
    save_scores(args, scores)

    if os.path.exists(args.cache_file_dir):
        import shutil
        shutil.rmtree(args.cache_file_dir)

if __name__ == "__main__":
    main()

