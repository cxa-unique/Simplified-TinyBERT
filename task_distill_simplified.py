"""
Distilling BERT (Simplified TinyBERT)
This script is modified over 'task_distill.py' in TinyBERT repository.
(https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import RandomSampler
from torch.nn import MSELoss

from bert.modeling import TinyBertForSequenceClassification
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam

from data_loader import distill_dataloader

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


def write_loss_to_file(loss_dict, file_name):
    with open(file_name, "a") as writer:
        for key in loss_dict.keys():
            writer.write("%s = %s\n" % (key, str(loss_dict[key])))
        writer.write("-----------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        required=True,
                        help="The GPU device you will run on.")
    parser.add_argument("--features_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The train features file. Should contain the .csv files (after tokenized) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir. Should contain the config/vocab/checkpoint file.")
    parser.add_argument("--general_student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model (after general distillation) dir. Should contain the config/vocab/checkpoint file.")
    parser.add_argument("--output_student_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the task-specific distilled student models will be output.")
    parser.add_argument("--cache_file_dir",
                        default='./cache',
                        type=str,
                        required=True,
                        help="The directory where cache the features.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-2,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--train_loss_step',
                        type=int,
                        default=1000,
                        help="How many train step to record a training loss.  ")
    parser.add_argument('--save_model_step',
                        type=int,
                        default=3000,
                        help="How many train step to save a student model.")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.,
                        help="The temperature in soft loss.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # Prepare device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_student_dir) and os.listdir(args.output_student_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_student_dir))
    if not os.path.exists(args.output_student_dir):
        os.makedirs(args.output_student_dir)
    if not os.path.exists(args.cache_file_dir):
        os.makedirs(args.cache_file_dir)

    # For save vocab file for all output models.
    tokenizer = BertTokenizer.from_pretrained(args.general_student_model, do_lower_case=args.do_lower_case)

    # Model
    teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=2)
    if args.fp16:
        teacher_model.half()
    teacher_model.to(device)

    student_model = TinyBertForSequenceClassification.from_pretrained(args.general_student_model, num_labels=2)
    student_model.to(device)

    # Train Config
    num_examples, train_dataloader = distill_dataloader(args, RandomSampler, batch_size=args.train_batch_size)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    num_train_optimization_steps = int(
        num_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    logger.info("***** Running Distilling *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    size = 0
    for n, p in student_model.named_parameters():
        logger.info('n: {}'.format(n))
        size += p.nelement()

    logger.info('Total parameters of student_model: {}'.format(size))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                         schedule=schedule,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=args.fp16_opt_level)
        logger.info('FP16 is activated, use amp')
    else:
        logger.info('FP16 is not activated, only use BertAdam')

    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    # Prepare loss functions
    loss_mse = MSELoss()

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    # Train
    global_step = 0
    output_loss_file = os.path.join(args.output_student_dir, "train_loss.txt")
    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        student_model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch
            if input_ids.size()[0] != args.train_batch_size:
                continue

            att_loss = 0.
            rep_loss = 0.

            student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask, is_student=True)
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)
            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                att_loss += tmp_loss

            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
            new_student_reps = student_reps
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss

            tr_att_loss += att_loss.item()
            tr_rep_loss += rep_loss.item()

            soft_loss = soft_cross_entropy(student_logits / args.temperature,
                                          teacher_logits / args.temperature)
            hard_loss = torch.nn.functional.cross_entropy(student_logits, label_ids, reduction='mean')
            cls_loss = soft_loss + hard_loss

            tr_cls_loss += cls_loss.item()

            loss = rep_loss + att_loss + cls_loss

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.train_loss_step == 0:
                loss = tr_loss / args.train_loss_step
                cls_loss = tr_cls_loss / args.train_loss_step
                att_loss = tr_att_loss / args.train_loss_step
                rep_loss = tr_rep_loss / args.train_loss_step

                loss_dict = {}
                loss_dict['global_step'] = global_step
                loss_dict['cls_loss'] = cls_loss
                loss_dict['att_loss'] = att_loss
                loss_dict['rep_loss'] = rep_loss
                loss_dict['loss'] = loss

                write_loss_to_file(loss_dict, output_loss_file)

                tr_loss = 0.
                tr_att_loss = 0.
                tr_rep_loss = 0.
                tr_cls_loss = 0.

            if global_step % args.save_model_step == 0:
                logger.info("***** Save model *****")

                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                model_name = WEIGHTS_NAME
                checkpoint_name = 'checkpoint-' + str(global_step)
                output_model_dir = os.path.join(args.output_dir, checkpoint_name)
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)
                output_model_file = os.path.join(output_model_dir, model_name)
                output_config_file = os.path.join(output_model_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_model_dir)

    if os.path.exists(args.cache_file_dir):
        import shutil
        shutil.rmtree(args.cache_file_dir)

if __name__ == "__main__":
    main()
