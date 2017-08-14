package ohs.ir.search.app;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class ConceptWeightEstimator2 {

	class PComparator implements Comparator<IntPair> {

		@Override
		public int compare(IntPair o1, IntPair o2) {
			int ret = o1.getFirst() - o2.getFirst();

			return ret;
		}
	}

	class Worker implements Callable<CounterMap<Integer, Integer>> {

		private ListMap<Integer, Integer> ii;

		private AtomicInteger row_cnt;

		private SparseMatrix T;

		private Timer timer;

		public Worker(SparseMatrix T, AtomicInteger range_cnt, ListMap<Integer, Integer> ii, Timer timer) {
			this.T = T;
			this.row_cnt = range_cnt;
			this.ii = ii;
			this.timer = timer;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {
			int m = 0;

			CounterMap<Integer, Integer> cm = Generics.newCounterMap();

			while ((m = row_cnt.getAndIncrement()) < T.rowSize()) {
				int p1 = T.indexAt(m);
				SparseVector t1 = T.rowAt(m);

				double avg1 = VectorMath.mean(t1);

				List<Integer> ns = getLocations(t1, m);

				Counter<Integer> c = cm.getCounter(p1);

				for (int n : ns) {
					if (n > m) {
						int p2 = T.indexAt(n);
						SparseVector t2 = T.rowAt(n);
						double avg2 = VectorMath.mean(t2);
						double cosine = VectorMath.cosine(t1, t2);
						cosine = Math.max(0, cosine);

						if (cosine > 0) {
							double score = cosine * avg1 * avg2;
							c.setCount(p2, score);
						}
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

		Counter<String> phrsCnts = Generics.newCounter();

		{
			Counter<String> docFreqs = Generics.newCounter();
			Counter<String> tfidfs = Generics.newCounter();

			for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt")) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				double tfidf = Double.parseDouble(ps[1]);
				double cnt = Double.parseDouble(ps[2]);
				double doc_freq = Double.parseDouble(ps[3]);
				docFreqs.setCount(phrs, doc_freq);
				tfidfs.setCount(phrs, tfidf);
			}

			Counter<String> c2 = FileUtils.readStringCounterFromText(MIRPath.DATA_DIR + "phrs/phrs_sorted.txt");

			for (String s : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_filtered.txt")) {
				String[] ps = s.split("\t");
				String phrs = ps[0];
				int rsc_cnt = Integer.parseInt(ps[1]);
				double doc_freq = docFreqs.getCount(phrs);
				double tfidf = tfidfs.getCount(phrs);

				if (doc_freq > 500) {
					phrsCnts.setCount(phrs, tfidf * rsc_cnt);
				}
			}

			System.out.printf("size: %d->%d\n", c2.size(), phrsCnts.size());
		}

		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_PM_2017_COL_MEDLINE_DC_DIR,
				MIRPath.STOPWORD_INQUERY_FILE);
		// ds.setScorer(new MarkovRandomFieldsScorer(ds));

		List<String> phrss = phrsCnts.getSortedKeys();

		Counter<String> c = Generics.newCounter();

		for (int i = 0; i < phrss.size(); i++) {
			String phrs = phrss.get(i);
			SparseVector scores = ds.search(phrs);
			// System.out.printf("%s, %f\n", phrs, scores.sum());

			int prog = BatchUtils.progress(i + 1, phrss.size());

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d]\n", prog, i + 1, phrss.size());
			}

			c.setCount(phrs, scores.sum());
		}

		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/phrs_score.txt", c);

		// ConceptWeightEstimator2 pre = new ConceptWeightEstimator2(vocab, wf,
		// phrsCnts, clueWords, wordToLemma, abbrMap);
		// pre.setThreadSize(10);
		// pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		System.out.println("process ends.");
	}

	private SetMap<String, String> abbrMap;

	private Counter<String> clueWords;

	private Counter<String> phrsCnts;

	private Indexer<String> phrsIdxer;

	private int thread_size = 5;

	private Vocab vocab;

	private WordFilter wf;

	private Indexer<String> wordIdxer;

	private Map<Integer, Integer> wordToLemma;

	private DenseVector wordWeights;

	public ConceptWeightEstimator2(Vocab vocab, WordFilter wf, Counter<String> phrsCnts, Counter<String> clueWords,
			Map<Integer, Integer> wordToLemma, SetMap<String, String> abbrMap) {
		this.vocab = vocab;
		this.wf = wf;
		this.phrsCnts = phrsCnts;
		this.clueWords = clueWords;
		this.wordToLemma = wordToLemma;
		this.abbrMap = abbrMap;

		phrsIdxer = Generics.newIndexer(phrsCnts.keySet());
	}

	private void computeWordWeights() {
		Counter<String> c = Generics.newCounter();

		for (String phrs : phrsIdxer.getObjects()) {
			for (String word : StrUtils.split(phrs)) {
				if (wf.filter(word)) {
					continue;
				}
				c.incrementCount(word, 1);
			}
		}

		wordIdxer = Generics.newIndexer(c.keySet());

		// System.out.println(cm1.toString());

		wordWeights = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			String word = wordIdxer.getObject(w);
			double cnt = c.getCount(word);
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
			wordWeights.add(w, tfidf);
		}
	}

	public Counter<String> estimate(String outFileName) throws Exception {

		computeWordWeights();

		SparseMatrix T = getSimilarityMatrix(getFeatureMatrix());

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = getPhraseBiases();

		VectorMath.randomWalk(T, cents, biases, 500, 0.000001, 0.85);

		Counter<String> ret = Generics.newCounter();

		for (int w = 0; w < phrsIdxer.size(); w++) {
			String phrs = phrsIdxer.getObject(w);
			double c = cents.value(w);
			if (c > 0) {
				c = Math.log(c);
				ret.incrementCount(phrs, c);
			}
		}

		FileUtils.writeStringCounterAsText(outFileName, ret);

		// System.out.println(ret.toStringSortedByValues(true, true, 100,
		// "\t"));

		return ret;
	}

	private SparseMatrix getFeatureMatrix() {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			int p = phrsIdxer.indexOf(phrs);
			if (p < 0) {
				continue;
			}

			List<String> words = StrUtils.split(phrs);
			for (String word : words) {
				int w = wordIdxer.indexOf(word);
				if (w < 0) {
					continue;
				}

				cm.incrementCount(p, w, 1);

				if (wordToLemma != null) {
					Integer l = wordToLemma.get(w);
					if (l != null) {
						cm.incrementCount(p, l, 1);
					}
				}
			}
		}

		for (int p : cm.keySet()) {
			Counter<Integer> c = cm.getCounter(p);
			for (Entry<Integer, Double> e : c.entrySet()) {
				int w = e.getKey();
				double tfidf = wordWeights.value(w);
				cm.setCount(p, w, tfidf);
			}
		}

		SparseMatrix F = new SparseMatrix(cm);
		// VectorMath.unitVector(F);
		return F;
	}

	private DenseVector getPhraseBiases() {
		DenseVector ret = new DenseVector(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			double weight = 0;
			for (String word : StrUtils.split(phrs)) {
				// weight += clueWords.getCount(word);
				int w = wordIdxer.indexOf(word);
				if (w < 0) {
					continue;
				}
				double cnt = clueWords.getCount(word);
				if (cnt > 0) {
					weight += wordWeights.value(w);
				}
			}

			double rsc_cnt = phrsCnts.getCount(phrs);

			ret.add(i, rsc_cnt);
		}

		ret.normalize();

		return ret;
	}

	private SparseMatrix getSimilarityMatrix(SparseMatrix X) throws Exception {
		Timer timer = Timer.newTimer();

		ListMap<Integer, Integer> ii = Generics.newListMap(X.rowSize());

		for (int m = 0; m < X.rowSize(); m++) {
			SparseVector row = X.rowAt(m);
			for (int n = 0; n < row.size(); n++) {
				int j = row.indexAt(n);
				ii.put(j, m);
			}
		}
		ii.trimToSize();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger row_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(X, row_cnt, ii, timer)));
		}

		CounterMap<Integer, Integer> cm = Generics.newCounterMap();

		for (int i = 0; i < thread_size; i++) {
			CounterMap<Integer, Integer> tmp = fs.get(i).get();
			cm.incrementAll(tmp);
		}
		tpe.shutdown();

		ListMap<Integer, Integer> ItoJs = Generics.newListMap(ListType.LINKED_LIST);

		for (int i : cm.keySet()) {
			for (int j : cm.getCounter(i).keySet()) {
				ItoJs.put(i, j);
			}
		}

		CounterMap<Integer, Integer> ret = Generics.newCounterMap(cm.size());

		for (int i : Generics.newArrayList(cm.keySet())) {
			Counter<Integer> c = cm.removeKey(i);
			for (Entry<Integer, Double> e : c.entrySet()) {
				int j = e.getKey();
				double v = e.getValue();
				ret.incrementCount(i, j, v);
				ret.incrementCount(j, i, v);
			}
		}

		// CounterMap<Integer, Integer> ret = Generics.newCounterMap(cm.size());
		//
		// for (int i : ItoJs.keySet()) {
		// LinkedList<Integer> js = (LinkedList<Integer>) ItoJs.get(i);
		// Counter<Integer> c = cm.getCounter(i);
		// for (int j : js) {
		// if (i != j) {
		// double v = c.getCount(j);
		// ret.incrementCount(i, j, v);
		// ret.incrementCount(j, i, v);
		// }
		// }
		// js.clear();
		// c.clear();
		//
		// c = null;
		// js = null;
		// }

		SparseMatrix T = new SparseMatrix(ret);
		T.normalizeColumns();

		return T;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
