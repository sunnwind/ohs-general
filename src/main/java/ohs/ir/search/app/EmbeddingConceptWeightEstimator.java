package ohs.ir.search.app;

import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class EmbeddingConceptWeightEstimator {

	class Worker implements Callable<CounterMap<Integer, Integer>> {

		private CounterMap<Integer, Integer> cm;

		private ListMap<Integer, Integer> ii;

		private double min_cosine = 0.3;

		private AtomicInteger row_cnt;

		private SparseMatrix T;

		private Timer timer;

		public Worker(SparseMatrix T, ListMap<Integer, Integer> ii, double min_cosine, AtomicInteger range_cnt,
				CounterMap<Integer, Integer> cm, Timer timer) {
			this.T = T;
			this.ii = ii;
			this.min_cosine = min_cosine;
			this.row_cnt = range_cnt;
			this.cm = cm;
			this.timer = timer;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {
			int m = 0;

			while ((m = row_cnt.getAndIncrement()) < T.rowSize()) {
				int p1 = T.indexAt(m);
				SparseVector t1 = T.rowAt(m);

				double avg1 = VectorMath.mean(t1);

				List<Integer> ns = getLocations(t1, m);

				Counter<Integer> c = null;

				synchronized (cm) {
					c = cm.getCounter(p1);
				}

				for (int n : ns) {
					if (n > m) {
						int p2 = T.indexAt(n);
						SparseVector t2 = T.rowAt(n);
						double avg2 = VectorMath.mean(t2);
						double cosine = VectorMath.cosine(t1, t2);

						if (cosine < min_cosine || cosine <= 0) {
							continue;
						}

						double score = cosine * avg1 * avg2;
						c.setCount(p2, score);
					}
				}

				int prog = BatchUtils.progress(m, T.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, m, T.rowSize(), timer.stop());
				}
			}

			return cm;
		}

		private List<Integer> getLocations(SparseVector a, int m) {
			Set<Integer> set = Generics.newHashSet();
			for (int j : a.indexes()) {
				for (int n : ii.get(j)) {
					if (n > m) {
						set.add(n);
					}
				}
			}
			List<Integer> ns = Generics.newArrayList(set);
			Collections.sort(ns);
			return ns;
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Indexer<String> phrsIdxer = null;
		DenseVector phrsBiases = null;

		{
			Counter<String> c = FileUtils.readStringCounterFromText(MIRPath.PHRS_DIR + "phrs_bias.txt");
			phrsIdxer = Generics.newIndexer(c.getSortedKeys());
			phrsBiases = VectorUtils.toDenseVector(c, phrsIdxer);

			// for (int i = 0; i < phrsBiases.size(); i++) {
			// double v = phrsBiases.value(i);
			// if (v < 0.5) {
			// phrsBiases.set(i, 0);
			// }
			// }

			VectorMath.exp(phrsBiases);

			phrsBiases.normalizeAfterSummation();
		}

		SparseMatrix S = null;

		{
			DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);

			Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
			WordFilter wf = new WordFilter(dc.getVocab(), stopwords);
			Vocab vocab = dc.getVocab();

			DenseVector phrsWeights = new DenseVector(phrsIdxer.size());

			for (int p = 0; p < phrsIdxer.size(); p++) {
				String phrs = phrsIdxer.getObject(p);
				List<String> words = StrUtils.split(phrs);
				double weight = 0;

				for (String word : words) {
					int w = vocab.indexOf(word);

					if (wf.filter(w)) {
						continue;
					}

					double cnt = vocab.getCount(word);
					double doc_freq = vocab.getDocFreq(word);

					if (doc_freq == 0 || cnt == 0) {
						continue;
					}

					double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);
					weight += tfidf;
				}
				phrsWeights.add(p, weight);
			}

			CounterMap<String, String> cm = FileUtils.readStringCounterMapFromText(MIRPath.PHRS_DIR + "phrs_sim.txt");
			CounterMap<Integer, Integer> cm2 = Generics.newCounterMap(cm.size());

			for (String phrs1 : Generics.newArrayList(cm.keySet())) {
				int p1 = phrsIdxer.indexOf(phrs1);

				if (p1 < 0) {
					continue;
				}

				double b1 = phrsWeights.value(p1);

				Counter<String> c = cm.removeKey(phrs1);

				for (Entry<String, Double> e : c.entrySet()) {
					String phrs2 = e.getKey();
					double cosine = e.getValue();
					int p2 = phrsIdxer.indexOf(phrs2);

					if (p2 < 0) {
						continue;
					}

					double b2 = phrsWeights.value(p2);

					// if (cosine < 0.9) {
					// continue;
					// }

					cosine = b1 * b2 * cosine;

					cm2.setCount(p1, p2, cosine);
					cm2.setCount(p2, p1, cosine);
				}
			}
			S = new SparseMatrix(cm2);
		}

		S.normalizeColumns();

		EmbeddingConceptWeightEstimator pre = new EmbeddingConceptWeightEstimator(phrsIdxer, S, phrsBiases);
		pre.setThreadSize(10);
		pre.setMinCosine(0.2);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		System.out.println("process ends.");
	}

	private DenseVector B;

	private double min_cosine = 0.3;

	private Indexer<String> phrsIdxer;

	private SparseMatrix S;

	private int thread_size = 5;

	public EmbeddingConceptWeightEstimator(Indexer<String> phrsIdxer, SparseMatrix S, DenseVector B) {
		this.phrsIdxer = phrsIdxer;
		this.S = S;
		this.B = B;

	}

	public Counter<String> estimate(String outFileName) throws Exception {

		DenseVector P = new DenseVector(phrsIdxer.size());

		// System.out.println(VectorUtils.toCounter(B,
		// phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		System.out.println("run random-walk");

		VectorMath.randomWalk(S, P, B, 500, 0.000001, 0.85, thread_size);

		Counter<String> ret = Generics.newCounter();

		for (int w = 0; w < phrsIdxer.size(); w++) {
			String phrs = phrsIdxer.getObject(w);
			double c = P.value(w);
			if (c > 0) {
				c = Math.log(c);
				ret.incrementCount(phrs, c);
			}
		}

		FileUtils.writeStringCounterAsText(outFileName, ret);

		// System.out.println(ret.toStringSortedByValues(true, true, 100, "\t"));

		return ret;
	}

	public void setMinCosine(double min_cosine) {
		this.min_cosine = min_cosine;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
