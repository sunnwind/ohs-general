package ohs.ir.search.model;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.Posting;
import ohs.ir.search.index.PostingList;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Vocab;
import ohs.utils.Timer;

public class VSMScorer extends Scorer {

	public static DenseVector getDocNorms(DocumentCollection dc) throws Exception {
		Timer timer = Timer.newTimer();

		DenseVector ret = new DenseVector(dc.size());
		int[][] rs = BatchUtils.getBatchRanges(dc.size(), 200);

		Vocab vocab = dc.getVocab();

		for (int k = 0; k < rs.length; k++) {
			int[] r = rs[k];
			SparseMatrix dvs = dc.getDocVectorRange(r[0], r[1]);
			for (int i = 0; i < dvs.rowSize(); i++) {
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

			int prog = BatchUtils.progress(k + 1, rs.length);

			if (prog > 0) {
				System.out.printf("[%d percent, %s]\n", prog, timer.stop());
			}
		}

		return ret;
	}

	private DenseVector docNorms;

	public VSMScorer(DocumentSearcher ds, DenseVector docNorms) {
		super(ds);
		this.docNorms = docNorms;
	}

	public VSMScorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii, DenseVector docNorms) {
		super(vocab, dc, ii);
		this.docNorms = docNorms;
	}

	@Override
	public void postprocess(SparseVector scores) {
		for (int i = 0; i < scores.size(); i++) {
			double score = scores.valueAt(i);
			score = Math.exp(score);
			scores.setAt(i, score);
		}
		scores.summation();
		scores.sortValues();
	}

	public void score(int w, double tfidf_w_in_q, PostingList pl, SparseVector ret) {
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
						double tfidf_w_in_d = TermWeighting.tfidf(cnt_w_in_d, vocab.getDocCnt(), vocab.getDocFreq(w));
						ret.addAt(m, tfidf_w_in_q * tfidf_w_in_d);
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
	public SparseVector score(SparseVector Q, SparseVector docs, boolean use_inverted_index) throws Exception {
		return null;
	}

	public SparseVector scoreFromCollection(SparseVector Q, SparseVector docCnts) throws Exception {
		SparseVector ret = null;

		return ret;
	}

	public SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));

		double norm_q = 0;

		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double cnt_w_in_q = Q.valueAt(i) + 1;
			double doc_freq = vocab.getDocFreq(w);
			double tfidf_w_in_q = TermWeighting.tfidf(cnt_w_in_q, dc.size(), doc_freq);

			norm_q += tfidf_w_in_q * tfidf_w_in_q;

			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}

			score(w, tfidf_w_in_q, pl, ret);

		}

		norm_q = Math.sqrt(norm_q);

		for (int i = 0; i < ret.size(); i++) {
			int dseq = ret.indexAt(i);
			double score = ret.valueAt(i);
			double norm_d = docNorms.value(dseq);
			if (norm_q > 0 && norm_d > 0) {
				score /= (norm_q * norm_d);
				ret.setAt(i, score);
			}
		}

		return ret;
	}

}
