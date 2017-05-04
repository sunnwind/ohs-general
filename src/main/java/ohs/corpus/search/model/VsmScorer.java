package ohs.corpus.search.model;

import java.util.Map.Entry;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.PostingList;
import ohs.corpus.type.DocumentCollection;
import ohs.ir.weight.TermWeighting;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class VsmScorer extends Scorer {

	public static DenseVector getDocNorms(DocumentCollection dc) throws Exception {
		DenseVector ret = new DenseVector(dc.size());
		int[][] rs = BatchUtils.getBatchRanges(dc.size(), 200);

		Vocab vocab = dc.getVocab();

		for (int k = 0; k < rs.length; k++) {
			int[] r = rs[k];
			SparseMatrix dvs = dc.getDocVectors(r[0], r[1]);
			for (int i = 0; k < dvs.rowSize(); i++) {
				int dseq = dvs.indexAt(i);
				SparseVector dv = dvs.rowAt(i);
				double norm = 0;
				for (int j = 0; j < dv.size(); j++) {
					int w = dv.indexAt(j);
					double cnt = dv.valueAt(j);
					double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
					norm += (tfidf * tfidf);
				}
				norm = Math.sqrt(norm);
				ret.add(dseq, norm);
			}
		}
		return ret;
	}

	private DenseVector docNorms;

	public VsmScorer(DocumentSearcher ds, DenseVector docNorms) {
		super(ds);
		this.docNorms = docNorms;
	}

	@Override
	public SparseVector score(SparseVector Q, SparseVector docCnts) throws Exception {
		Q = Q.copy();
		Counter<Integer> scores = Generics.newCounter();

		double norm_q = 0;

		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double cnt_w_in_q = Q.valueAt(i) + 1;

			double doc_freq = vocab.getDocFreq(w);

			double tfidf_w_in_q = TermWeighting.tfidf(cnt_w_in_q, dc.size(), doc_freq);

			norm_q += tfidf_w_in_q * tfidf_w_in_q;

			PostingList p = ii.getPostingList(w);

			if (p == null) {
				continue;
			}

			IntegerArray dseqs = p.getDocSeqs();
			IntegerArray sizes = p.getCounts();

			for (int j = 0; j < dseqs.size(); j++) {
				int dseq = dseqs.get(j);
				double cnt_w = sizes.get(j);
				double tfidf_w_in_d = TermWeighting.tfidf(cnt_w, dc.size(), doc_freq);
				scores.incrementCount(dseq, tfidf_w_in_q * tfidf_w_in_d);
			}
		}

		norm_q = Math.sqrt(norm_q);

		SparseVector ret = new SparseVector(scores.size());
		int loc = 0;

		for (Entry<Integer, Double> e : scores.entrySet()) {
			int dseq = e.getKey();
			double score = e.getValue();
			double norm_d = docNorms.value(dseq);
			score /= (norm_q * norm_d);
			ret.addAt(loc, dseq, score);
			loc++;
		}
		return ret;
	}

}
