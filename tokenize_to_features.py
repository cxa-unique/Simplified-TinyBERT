import logging
from bert.tokenization import BertTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(set_name, examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if set_name == "train":
            label_id = example.label
        elif set_name == "dev" or set_name == "test":
            label_id = '0'   # unnecessary for dev/test examples
        else:
            raise KeyError(set_name)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label id = %s" % label_id)

        features.append(
            InputFeatures(guid=example.guid,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def tokenize_to_features(set_name, raw_data_file, output_features_file, tokenizer, max_seq_length):

    examples = []
    with open(raw_data_file, 'r') as raw_file:
        for line in raw_file:
            example_id, query_text, passage_text, label = line.strip().split('\t')
            examples.append(InputExample(example_id, query_text, passage_text, label))
    features = convert_examples_to_features(set_name, examples, max_seq_length, tokenizer)

    with open(output_features_file, 'w') as csv_file:
        csv_file.write('example_id,input_ids,input_mask,segment_ids,label' + '\n')
        for i, f in enumerate(features):
            input_ids_str = " ".join([str(id) for id in f.input_ids])
            input_mask_str = " ".join([str(id) for id in f.input_mask])
            segment_ids_str = " ".join([str(id) for id in f.segment_ids])
            label = str(f.label_id)
            csv_file.write(f.guid + ',' + input_ids_str + ',' + input_mask_str + ',' + segment_ids_str + ',' + label + '\n')


def main():

    ## data
    set_name = 'test' # 'train', 'dev' or 'test'
    raw_data_file = 'PAIRS_FILE'  # contains query passage pairs, format: example_id \t query_text \t passage text (\t label) \n
    output_features_file = 'FEATURES_FILE'

    ## prepare tokenizer
    bert_model_dir = 'BERT_MODEL_DIR' # contains vocab.txt file.
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
    max_seq_length = 256

    # start tokenize
    tokenize_to_features(set_name, raw_data_file, output_features_file, tokenizer, max_seq_length)
    logger.info('Convert to csv done!')

if __name__ == "__main__":
    main()
