package ohs.corpus.search.model;

import ohs.corpus.search.index.WordFilter;
import ohs.corpus.type.DocumentCollection;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class DocumentPriorEstimator {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		System.out.println("process ends.");
	}

	private double prior_dir = 2000;

	private double mixture_jm = 0;

	private DocumentCollection dc;

	private Vocab vocab;

	private Counter<Integer> cache;

	private WordFilter filter;

	public DocumentPriorEstimator(String idxDir) throws Exception {
		dc = new DocumentCollection(idxDir);
		vocab = dc.getVocab();
	}

	public DocumentPriorEstimator(Vocab vocab, DocumentCollection ds, WordFilter filter) {
		this.vocab = vocab;
		this.dc = ds;
		this.filter = filter;
		cache = Generics.newCounter();
	}

	public DenseVector estimate() throws Exception {
		return estimateUsingLM(ArrayUtils.range(dc.size())).toDenseVector();
	}

	public SparseVector estimateUsingDocLen(int[] dseqs) throws Exception {
		SparseVector ret = new SparseVector(dseqs.length);

		for (int i = 0; i < dseqs.length; i++) {
			int dseq = dseqs[i];
			double prior = 0;

			if (cache.containsKey(dseq)) {
				prior = cache.getCount(dseq);
			} else {
				SparseVector dv = dc.getDocVector(dseq);
				for (int j = 0; j < dv.size(); j++) {
					int w = dv.indexAt(j);

					if (filter.filter(w)) {
						continue;
					}

					double cnt_w = dv.valueAt(j);
					prior += cnt_w;
				}
				cache.setCount(dseq, prior);
			}
			ret.addAt(i, dseq, prior);
		}
		return ret;
	}

	public SparseVector estimateUsingIDF(int[] dseqs) throws Exception {
		SparseVector ret = new SparseVector(dseqs.length);

		for (int i = 0; i < dseqs.length; i++) {
			int dseq = dseqs[i];
			double prior = 0;

			if (cache.containsKey(dseq)) {
				prior = cache.getCount(dseq);
			} else {
				SparseVector dv = dc.getDocVector(dseq);
				double size = 0;
				for (int j = 0; j < dv.size(); j++) {
					int w = dv.indexAt(j);

					if (filter.filter(w)) {
						continue;
					}

					double cnt_w = dv.valueAt(j);
					double doc_freq = vocab.getDocFreq(w);
					double num_docs = vocab.getDocCnt();
					// double tfidf = TermWeighting.tfidf(cnt_w, num_docs,
					// doc_freq);
					double idf = TermWeighting.idf(num_docs, doc_freq);
					prior += idf;
					size++;
				}
				// prior /= size;
				cache.setCount(dseq, prior);
			}
			ret.addAt(i, dseq, prior);
		}

		return ret;
	}

	public SparseVector estimateUsingLM(int[] dseqs) throws Exception {
		SparseVector ret = new SparseVector(dseqs.length);

		for (int i = 0; i < dseqs.length; i++) {
			int dseq = dseqs[i];
			double prior = 0;

			if (cache.containsKey(dseq)) {
				prior = cache.getCount(dseq);
			} else {
				SparseVector dv = dc.getDocVector(dseq);
				// double[] log_probs = new double[dv.size()];

				for (int j = 0; j < dv.size(); j++) {
					int w = dv.indexAt(j);

					double cnt_w_in_d = dv.valueAt(j);
					double len_d = dv.sum();
					double pr_w_in_c = vocab.getProb(w);
					double pr_w_in_d_jm = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir,
							pr_w_in_c, mixture_jm);
					prior += pr_w_in_d_jm;
					// log_probs[j] = Math.log(pr_w_in_d_jm);
				}

				// prior = ArrayMath.sumLogProbs(log_probs);
				// prior = Math.exp(prior);

				cache.setCount(dseq, prior);
			}

			ret.addAt(i, dseq, prior);
		}

		return ret;
	}

	public double getDirichletPrior() {
		return prior_dir;
	}

	public double getMixtureJM() {
		return mixture_jm;
	}

	public void setDirichletPrior(double dirichlet_prior) {
		this.prior_dir = dirichlet_prior;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}
}
