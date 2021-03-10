import heapq
import numpy as np


def save_trec_results(scores_file, output_results_file, aggregation, run_name):
    rankings = {}
    with open(scores_file, mode='r') as scores_file:
        for line in scores_file:
            q_id, p_id, s = line.strip().split('\t')

            doc_id = p_id.split('_')[0]
            if aggregation == 'FirstP' and p_id.split('-')[1] != '0':
                continue
            score = float(s)
            if q_id not in rankings:
                rankings[q_id] = {}
            if doc_id not in rankings[q_id]:
                rankings[q_id][doc_id] = []
            rankings[q_id][doc_id].append((score, p_id))

    with open(output_results_file, 'w') as out_file:
        for qid in rankings:
            my_ranking = []
            for docid in rankings[qid]:
                scores = [s for s, _ in rankings[qid][docid]]
                passages = [p for _, p in rankings[qid][docid]]
                if aggregation == "MaxP":
                    max_score_idx = np.argmax(scores)
                    max_score = scores[max_score_idx]
                    my_ranking.append((max_score, docid))
                elif aggregation == "SumP":
                    sum_score = np.sum(scores)
                    my_ranking.append((sum_score, docid))
                elif aggregation == "AvgP":
                    avg_score = np.sum(scores) / len(scores)
                    my_ranking.append((avg_score, docid))
                elif aggregation == "FirstP":
                    assert len(scores) == 1
                    my_ranking.append((scores[0], docid))
                elif aggregation.split('-')[1] == "Max&AvgP":
                    k = int(aggregation.split('-')[0])
                    max_index_list = list(map(scores.index, heapq.nlargest(k, scores)))
                    if len(max_index_list) < k:
                        print(max_index_list)
                    ksum_score = 0.0
                    for idx in max_index_list:
                        ksum_score += scores[idx]
                    ksum_score = ksum_score / len(max_index_list)
                    my_ranking.append((ksum_score, docid))
                else:
                    raise NotImplementedError

            sorted_ranking = sorted(my_ranking, reverse=True)
            for rank, item in enumerate(sorted_ranking):
                score, docid = item
                out_str = "{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(qid, docid, rank + 1, score, run_name)
                out_file.write(out_str)

def main():
    scores_file = 'PATH_TO_SCORE_FILE' ##Format: query_id \t passage_id \t score \n
    output_results_file = 'PATH_TO_TREC_RESULT_FILE' ##Format: TREC
    aggregation = 'MaxP'  # We use BERT-MaxP in our experiments.
    run_name = 'bert_' + aggregation

    save_trec_results(scores_file, output_results_file, aggregation, run_name)

if __name__ == "__main__":
    main()
