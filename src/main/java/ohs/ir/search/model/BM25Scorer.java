package ohs.ir.search.model;

import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.search.index.Posting;
import ohs.ir.search.index.PostingList;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.matrix.SparseVector;

public class BM25Scorer extends Scorer {

	private double k1 = TermWeighting.k1;

	private double k3 = TermWeighting.k3;

	private double b = TermWeighting.b;

	private double sigma = 0;

	private double num_docs = vocab.getDocCnt();

	private double len_d_avg = dc.getAvgDocLength();

	public BM25Scorer(DocumentSearcher ds) {
		super(ds);
	}

	public void score(int w, double cnt_w_in_q, PostingList pl, SparseVector ret) {
		int m = 0;
		int n = 0;

		while (m < ret.size() && n < pl.size()) {
			int dseq1 = ret.indexAt(m);
			Posting p = pl.getPosting(n);
			int dseq2 = p.getDocseq();

			if (ret.location(dseq2) == -1) {
				n++;
			} else {
				if (dseq1 == dseq2 || dseq1 < dseq2) {
					if (dseq1 == dseq2) {
						double cnt_w_in_d = dseq1 == dseq2 ? p.size() : 0;
						double len_d = dc.getDocLength(dseq1);
						double doc_freq = vocab.getDocFreq(w);
						double score = TermWeighting.bm25(cnt_w_in_q, k3, cnt_w_in_d, len_d, len_d_avg, num_docs, doc_freq, b, k1, sigma);
						ret.addAt(m, score);
						m++;
						n++;
					} else {
						m++;
					}
				} else {
					n++;
				}
			}
		}
	}

	@Override
	public SparseVector scoreFromCollection(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));

		for (int i = 0; i < docs.size(); i++) {
			int dseq = docs.indexAt(i);
			SparseVector dv = dc.getDocVector(dseq);
			double score = 0;
			for (int j = 0; j < Q.size(); j++) {
				int w = Q.indexAt(j);
				double cnt_w_in_q = Q.valueAt(j);
				double cnt_w_in_d = dv.value(w);
				double len_d = dv.sum();
				double doc_freq = vocab.getDocFreq(w);

				if (cnt_w_in_d > 0) {
					score += TermWeighting.bm25(cnt_w_in_q, k3, cnt_w_in_d, len_d, len_d_avg, num_docs, doc_freq, b, k1, sigma);
				}
			}
			ret.addAt(i, score);
		}
		return ret;
	}

	@Override
	public SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));
		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double cnt_w_in_q = Q.valueAt(i);

			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}
			score(w, cnt_w_in_q, pl, ret);
		}
		return ret;
	}

}
