package ohs.corpus.search.model;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.PostingList;
import ohs.ir.weight.TermWeighting;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class BM25Scorer extends Scorer {

	public BM25Scorer(DocumentSearcher ds) {
		super(ds);
	}

	@Override
	public SparseVector score(SparseVector Q, SparseVector docs) throws Exception {
		Counter<Integer> c = Generics.newCounter();

		double k1 = TermWeighting.k1;
		double k3 = TermWeighting.k3;
		double b = TermWeighting.b;
		double sigma = 0;
		double num_docs = vocab.getDocCnt();

		double len_d_avg = dc.getAvgDocLength();

		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double cnt_w_in_q = Q.valueAt(i);

			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}

			IntegerArray dseqs = pl.getDocSeqs();
			IntegerArray sizes = pl.getCounts();
			double doc_freq = vocab.getDocFreq(w);

			for (int j = 0; j < dseqs.size(); j++) {
				int dseq = dseqs.get(j);
				double cnt_w_in_d = sizes.get(j);
				double len_d = dc.getDocLength(dseq);
				double score = TermWeighting.bm25(cnt_w_in_q, k3, cnt_w_in_d, len_d, len_d_avg, num_docs, doc_freq, b, k1, sigma);
				c.incrementCount(dseq, score);
			}
		}
		SparseVector ret = new SparseVector(c);
		return ret;
	}

}
