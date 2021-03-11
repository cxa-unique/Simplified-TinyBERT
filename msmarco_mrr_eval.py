import sys

def cal_mrr(qrels_file, result_file):
    qrel = {}
    with open(qrels_file, 'r', encoding='utf8') as f:
        for line in f:
            topicid, _, docid, rel = line.strip().split()
            # assert rel == "1"
            if rel == '0':
                continue
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]


    results = {}
    with open(result_file, 'r') as file:
        for line in file:
            [qid, _, docid, rank, _, _] = line.split()

            if qid not in results:
                results[qid] = []

            results[qid].append([docid, int(rank)])


    query_num = 0
    mrr_score = 0.0
    mrr_ten_score = 0.0

    for query in results:

        assert query in qrel, 'query not in qrels!'

        query_num += 1

        for [doc, rank] in results[query]:
            if doc in qrel[query]:
                mrr_score += 1.0 / rank
                break

        for [doc, rank] in results[query][:10]:
            if doc in qrel[query]:
                mrr_ten_score += 1.0 / rank
                break

    mrr = mrr_score / query_num
    mrr_ten = mrr_ten_score / query_num

    return query_num, mrr, mrr_ten

def main():
    """Command line:
    python msmarco_mrr_eval.py <path to reference> <path_to_candidate_file>
    """
    path_to_candidate = sys.argv[2]
    path_to_reference = sys.argv[1]
    query_num, mrr, mrr_ten = cal_mrr(path_to_reference, path_to_candidate)
    print('#####################')
    print('query_num: ' + str(query_num))
    print('MRR: ' + str(mrr))
    print('MRR@10: ' + str(mrr_ten))
    print('#####################')


if __name__ == '__main__':
    main()


