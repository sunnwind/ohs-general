package ohs.ir.search.model;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class DocumentPriorEstimator {

	public enum Type {
		CDD, QDLM, IDF, DLM, LEN, STOP_RATIO
	}

	class Worker implements Callable<Integer> {

		private AtomicInteger range_cnt;

		private int[][] rs;

		private DocumentCollection dc;

		private Timer timer;

		private DenseVector priors;

		public Worker(DocumentCollection dc, int[][] ranges, AtomicInteger range_cnt, DenseVector priors, Timer timer) {
			this.dc = dc;
			this.rs = ranges;
			this.range_cnt = range_cnt;
			this.priors = priors;
			this.timer = timer;
		}

		@Override
		public Integer call() throws Exception {
			int loc = 0;

			while ((loc = range_cnt.getAndIncrement()) < rs.length) {
				int[] r = rs[loc];
				DenseVector subPriors = estimate(r[0], r[1]);

				for (int i = 0; i < subPriors.size(); i++) {
					int dseq = i + r[0];
					priors.set(dseq, subPriors.value(i));
				}

				int prog = BatchUtils.progress(loc, rs.length);
				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %d/%d, %s]\n", prog, loc, rs.length, r[1], dc.size(), timer.stop());
				}
			}
			return null;
		}

		@Override
		protected void finalize() throws Throwable {
			super.finalize();
			dc.close();
		}
	}

	private int batch_size = 1000;

	private int thread_size = 5;

	private double prior_dir = 2000;

	private double mixture_jm = 0;

	private DocumentCollection dc;

	private Vocab vocab;

	private WordFilter wf;

	private Type type = Type.CDD;

	private SparseVector Q;

	public DocumentPriorEstimator(DocumentCollection dc, WordFilter wf) {
		this.dc = dc;
		this.vocab = dc.getVocab();
		this.wf = wf;
	}

	public DocumentPriorEstimator(String idxDir) throws Exception {
		dc = new DocumentCollection(idxDir);
		vocab = dc.getVocab();
	}

	public DenseVector estimate() throws Exception {
		Timer timer = Timer.newTimer();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		List<DocumentCollection> dss = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			dss.add(dc.copyShallow());
		}

		int[][] rs = BatchUtils.getBatchRanges(dc.size(), batch_size);

		AtomicInteger range_cnt = new AtomicInteger(0);

		DenseVector priors = new DenseVector(dc.size());

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(dss.get(i), rs, range_cnt, priors, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		fs.clear();

		priors.summation();

		tpe.shutdown();

		for (int i = 0; i < thread_size; i++) {
			dss.get(i).close();
		}

		return priors;
	}

	public DenseVector estimate(int i, int j) throws Exception {
		DenseVector ret = null;
		if (type == Type.CDD) {
			ret = estimateCollectionDocumentDistancePriors(i, j);
		} else if (type == Type.QDLM) {
			ret = estimateQueryDependentPriors(i, j);
		} else if (type == Type.IDF) {
			ret = estimateIDFPriors(i, j);
		} else if (type == Type.DLM) {
			ret = estimateLanguageModelPriors(i, j);
		} else if (type == Type.LEN) {
			ret = estimateLengthPriors(i, j);
		} else if (type == Type.STOP_RATIO) {
			ret = estimateStopRatioPriors(i, j);
		}
		return ret;
	}

	public DenseVector estimateCollectionDocumentDistancePriors(int i, int j) throws Exception {
		DenseVector ret = new DenseVector(j - i);
		SparseMatrix dvs = dc.getDocVectorRange(i, j);

		for (int k = 0; k < dvs.rowSize(); k++) {
			int dseq = dvs.indexAt(k);
			SparseVector dv = dvs.rowAt(k);
			double div_sum = 0;

			for (int l = 0; l < dv.size(); l++) {
				int w = dv.indexAt(l);
				double cnt_w_in_d = dv.valueAt(l);
				double len_d = dv.sum();
				double pr_w_in_c = vocab.getProb(w);
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
				div_sum += pr_w_in_d * Math.log(pr_w_in_d / pr_w_in_c);
			}

			if (div_sum != 0) {
				double score = Math.exp(-div_sum);
				ret.add(k, score);
			}
		}
		return ret;
	}

	public DenseVector estimateIDFPriors(int i, int j) throws Exception {
		DenseVector ret = new DenseVector(j - i);
		SparseMatrix dvs = dc.getDocVectorRange(i, j);

		for (int k = 0; k < dvs.rowSize(); k++) {
			int dseq = dvs.indexAt(k);
			SparseVector dv = dvs.rowAt(k);
			double avg_idf = 0;
			for (int l = 0; l < dv.size(); l++) {
				int w = dv.indexAt(l);
				double doc_freq = vocab.getDocFreq(w);
				double idf = TermWeighting.idf(vocab.getDocCnt(), doc_freq);
				avg_idf += idf;
			}

			if (avg_idf > 0) {
				avg_idf /= dv.size();
				ret.add(k, avg_idf);
			}
		}
		return ret;
	}

	public DenseVector estimateStopRatioPriors(int i, int j) throws Exception {
		DenseVector ret = new DenseVector(j - i);
		SparseMatrix dvs = dc.getDocVectorRange(i, j);

		for (int k = 0; k < dvs.rowSize(); k++) {
			int dseq = dvs.indexAt(k);
			SparseVector dv = dvs.rowAt(k);

			double cnt_stop = 0;
			double cnt_nonstop = 0;

			for (int l = 0; l < dv.size(); l++) {
				int w = dv.indexAt(l);
				double cnt = dv.valueAt(l);

				if (wf.filter(w)) {
					cnt_stop += cnt;
				} else {
					cnt_nonstop += cnt;
				}
			}

			if (dv.sum() > 0) {
				double ratio = (cnt_nonstop + 1) / (cnt_stop + 1);
				ret.add(k, ratio);
			}
		}
		return ret;
	}

	public DenseVector estimateLanguageModelPriors(int i, int j) throws Exception {
		DenseVector ret = new DenseVector(j - i);
		SparseMatrix dvs = dc.getDocVectorRange(i, j);

		for (int k = 0; k < dvs.rowSize(); k++) {
			int dseq = dvs.indexAt(k);
			SparseVector dv = dvs.rowAt(k);
			double score = 0;

			for (int l = 0; l < dv.size(); l++) {
				int w = dv.indexAt(l);
				double cnt_w_in_d = dv.valueAt(l);
				double len_d = dv.sum();
				double pr_w_in_c = vocab.getProb(w);
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
				score += pr_w_in_d;
			}
			ret.add(k, score);
		}
		return ret;
	}

	public DenseVector estimateLengthPriors(int i, int j) throws Exception {
		DenseVector ret = new DenseVector(j - i);
		SparseMatrix dvs = dc.getDocVectorRange(i, j);
		double len_avg = dc.getAvgDocLength();
		for (int k = 0; k < dvs.rowSize(); k++) {
			int dseq = dvs.indexAt(k);
			SparseVector dv = dvs.rowAt(k);
			double len_d = dv.sum();
			double prior = len_d / len_avg;
			ret.add(k, prior);
		}
		return ret;
	}

	public DenseVector estimateQueryDependentPriors(int i, int j) throws Exception {
		DenseVector ret = new DenseVector(j - i);
		SparseMatrix dvs = dc.getDocVectorRange(i, j);

		for (int k = 0; k < dvs.rowSize(); k++) {
			int dseq = dvs.indexAt(k);
			SparseVector dv = dvs.rowAt(k);
			double score = 0;

			for (int l = 0; l < dv.size(); l++) {
				int w = dv.indexAt(l);
				if (Q.location(w) < 0) {
					continue;
				}
				double cnt_w_in_d = dv.value(w);
				double len_d = dv.sum();
				double pr_w_in_c = vocab.getProb(w);
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
				// div_sum += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
				score += pr_w_in_d;
			}
			ret.add(k, score);

			// if (div_sum != 0) {
			// double score = Math.exp(-div_sum);
			// ret.add(k, score);
			// }
		}
		return ret;
	}

	public double getDirichletPrior() {
		return prior_dir;
	}

	public double getMixtureJM() {
		return mixture_jm;
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setDirichletPrior(double dirichlet_prior) {
		this.prior_dir = dirichlet_prior;
	}

	public void setJmMixture(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setMixtureJM(double mixture_jm) {
		this.mixture_jm = mixture_jm;
	}

	public void setQuery(SparseVector Q) {
		this.Q = Q;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	public void setType(Type type) {
		this.type = type;
	}

}
