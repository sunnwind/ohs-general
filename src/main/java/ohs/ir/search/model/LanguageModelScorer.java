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
import ohs.types.generic.Vocab;

public class LanguageModelScorer extends Scorer {

	public static enum Type {
		QL, KLD
	}

	public static SparseVector score(SparseVector lm_q, SparseMatrix ds, Vocab vocab, double prior_dir, double mixture_jm) {
		SparseVector ret = new SparseVector(ds.size());
		for (int i = 0; i < ds.size(); i++) {
			ret.addAt(i, ds.indexAt(i), score(lm_q, ds.rowAt(i), vocab, prior_dir, mixture_jm));
		}
		return ret;
	}

	/**
	 * @param lm_q
	 * @param d
	 * @param vocab
	 * @param prior_dir
	 * @param mixture_jm
	 * @return
	 */
	public static double score(SparseVector lm_q, SparseVector d, Vocab vocab, double prior_dir, double mixture_jm) {
		double len_d = d.sum();
		double div = 0;

		for (int i = 0; i < lm_q.size(); i++) {
			int w = lm_q.indexAt(i);
			double pr_w_in_q = lm_q.probAt(i);
			double pr_w_in_c = vocab.getProb(w);
			double cnt_w_in_d = d.value(w);
			double pr_w_in_d_jm = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);

			if (pr_w_in_d_jm > 0) {
				div += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d_jm);
			}
		}

		double score = Math.exp(-div);
		return score;
	}

	protected Type type = Type.KLD;

	protected double prior_dir = 2000;

	protected double mixture_jm = 0.5;

	protected DenseVector lm_qbg;

	public LanguageModelScorer(DocumentSearcher ds) {
		super(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex());
	}

	public LanguageModelScorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii) {
		super(vocab, dc, ii);
	}

	public double getDirichletPrior() {
		return prior_dir;
	}

	public double getJmMixture() {
		return mixture_jm;
	}

	public DenseVector getQueryBackgroundModel() {
		return lm_qbg;
	}

	@Override
	public void postprocess(SparseVector scores) {
		if (type == Type.QL) {
			for (int i = 0; i < scores.size(); i++) {
				if (scores.valueAt(i) == 0) {
					scores.setAt(i, Double.NEGATIVE_INFINITY);
				}
			}
		} else if (type == Type.KLD) {
			for (int i = 0; i < scores.size(); i++) {
				int dseq = scores.indexAt(i);
				double div = scores.valueAt(i);
				double score = Math.exp(-div);
				scores.setAt(i, score);
			}
		}

		if (docPriors != null) {
			for (int i = 0; i < scores.size(); i++) {
				int w = scores.indexAt(i);
				double score = scores.valueAt(i);
				double doc_prior = docPriors.value(w);
				scores.setAt(i, score * doc_prior);
			}
		}

		scores.summation();
		scores.sortValues();
	}

	public void score(int w, double pr_w_in_q, PostingList pl, SparseVector ret) {
		double len_c = dc.getLength();
		double cnt_w_in_c = pl.getCount();
		double pr_w_in_c = cnt_w_in_c / len_c;
		double pr_w_in_qbg = pr_w_in_c;

		if (lm_qbg != null && w != -1) {
			pr_w_in_qbg = lm_qbg.value(w);
		}

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
					double cnt_w_in_d = dseq1 == dseq2 ? p.size() : 0;
					double len_d = dc.getDocLength(dseq1);
					double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_qbg, mixture_jm);

					if (pr_w_in_d > 0) {
						double val = 0;
						if (type == Type.KLD) {
							val = pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
						} else {
							val = Math.log(pr_w_in_d);
						}
						ret.addAt(m, val);
					}

					if (dseq1 == dseq2) {
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

		while (m < ret.size()) {
			int dseq = ret.indexAt(m);
			double cnt_w_in_d = 0;
			double len_d = dc.getDocLength(dseq);
			double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_qbg, mixture_jm);
			if (pr_w_in_d > 0) {
				double val = 0;
				if (type == Type.KLD) {
					val = pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
				} else {
					val = Math.log(pr_w_in_d);
				}
				ret.addAt(m, val);
			}
			m++;
		}
	}

	public SparseVector scoreFromCollection(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));

		for (int i = 0; i < docs.size(); i++) {
			int dseq = docs.indexAt(i);
			SparseVector dv = dc.getDocVector(dseq);
			double score = 0;

			for (int j = 0; j < Q.size(); j++) {
				int w = Q.indexAt(j);
				double pr_w_in_q = Q.probAt(j);
				String word = vocab.getObject(w);

				double len_c = dc.getLength();
				double cnt_w_in_c = vocab.getCount(w);
				double pr_w_in_c = cnt_w_in_c / len_c;
				double pr_w_in_qbg = pr_w_in_c;

				if (lm_qbg != null) {
					pr_w_in_qbg = lm_qbg.value(w);
				}

				double cnt_w_in_d = dv.value(w);
				double len_d = dv.sum();
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_qbg, mixture_jm);

				if (pr_w_in_d > 0) {
					double val = 0;
					if (type == Type.KLD) {
						val = pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
					} else {
						val = Math.log(pr_w_in_d);
					}
					score += val;
				}
			}
			ret.addAt(i, score);
		}
		return ret;
	}


	public SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));
		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double pr_w_in_q = Q.probAt(i);
			String word = vocab.getObject(w);

			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}
			// System.out.printf("word=[%s], %s\n", word, pl.toString());
			score(w, pr_w_in_q, pl, ret);
		}
		return ret;
	}

	public void setDirichletPrior(double prior_dir) {
		this.prior_dir = prior_dir;
	}

	public void setJmMixture(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setQueryBackgroundModel(DenseVector lm_qbg) {
		this.lm_qbg = lm_qbg;
	}

	public void setType(Type type) {
		this.type = type;
	}

}
