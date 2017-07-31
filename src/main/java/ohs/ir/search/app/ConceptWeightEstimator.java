package ohs.ir.search.app;

import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.TrecCdsQuery;
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
public class ConceptWeightEstimator {

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
			List<String> lines = FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt");
			phrsCnts = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				double cnt = Double.parseDouble(ps[2]);
				double doc_freq = Double.parseDouble(ps[3]);
				phrsCnts.setCount(phrs, doc_freq);
			}
		}

		int size1 = phrsCnts.size();
		phrsCnts.pruneKeysBelowThreshold(100);

		System.out.printf("size: %d->%d\n", size1, phrsCnts.size());

		Vocab vocab = DocumentCollection.readVocab(MIRPath.DATA_DIR + "merged/col/dc/vocab.ser");

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(vocab, stopwords);

		Counter<String> clueWords = Generics.newCounter();

		Counter<String> filteredCnts = Generics.newCounter();

		{
			StringTokenizer st = new EnglishTokenizer();

			Pattern puncPat = Pattern.compile("^\\p{Punct}+");

			for (BaseQuery bq : QueryReader.readQueries(MIRPath.TREC_PM_2017_QUERY_FILE)) {
				List<String> words = st.tokenize(bq.getSearchText());

				for (String word : words) {
					if (wf.filter(word)) {
						continue;
					}
					clueWords.incrementCount(word, 1);
				}
			}

			for (BaseQuery bq : QueryReader.readQueries(MIRPath.TREC_CDS_2014_QUERY_FILE)) {
				List<String> words = st.tokenize(bq.getSearchText());

				for (String word : words) {
					if (wf.filter(word)) {
						continue;
					}
					clueWords.incrementCount(word, 1);
				}
			}

			for (BaseQuery bq : QueryReader.readQueries(MIRPath.TREC_CDS_2015_QUERY_A_FILE)) {
				List<String> words = st.tokenize(bq.getSearchText());

				for (String word : words) {
					if (wf.filter(word)) {
						continue;
					}
					clueWords.incrementCount(word, 1);
				}
			}

			for (BaseQuery bq : QueryReader.readQueries(MIRPath.TREC_CDS_2016_QUERY_FILE)) {
				TrecCdsQuery tcq = (TrecCdsQuery) bq;
				List<String> l = Generics.newArrayList();
				l.add(tcq.getDescription());
				l.add(tcq.getSummary());
				l.add(tcq.getNote());

				String s = StrUtils.join("\n", l);

				List<String> words = st.tokenize(s);

				for (String word : words) {
					if (wf.filter(word)) {
						continue;
					}
					clueWords.incrementCount(word, 1);
				}
			}

			for (BaseQuery bq : QueryReader.readQueries(MIRPath.CLEF_EH_2016_QUERY_FILE)) {
				List<String> words = st.tokenize(bq.getSearchText());

				for (String word : words) {
					if (wf.filter(word)) {
						continue;
					}
					clueWords.incrementCount(word, 1);
				}
			}
		}

		ConceptWeightEstimator pre = new ConceptWeightEstimator(vocab, wf, phrsCnts, clueWords);
		pre.setThreadSize(10);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		System.out.println("process ends.");
	}

	private Counter<String> clueWords;

	private Counter<String> phrsCnts;

	private Indexer<String> phrsIdxer;

	private int thread_size = 5;

	private Vocab vocab;

	private WordFilter wf;

	private Indexer<String> wordIdxer;

	private DenseVector wordWeights;

	public ConceptWeightEstimator(Vocab vocab, WordFilter wf, Counter<String> phrsCnts, Counter<String> clueWords) {
		this.vocab = vocab;
		this.wf = wf;
		this.phrsCnts = phrsCnts;
		this.clueWords = clueWords;
	}

	private void computeWordWeights() {
		Counter<String> wordCnts = Generics.newCounter();

		for (String phrs : phrsIdxer.getObjects()) {
			for (String word : StrUtils.split(phrs)) {
				if (wf.filter(word)) {
					continue;
				}
				wordCnts.incrementCount(word, 1);
			}
		}

		wordIdxer = Generics.newIndexer(wordCnts.keySet());

		// System.out.println(cm1.toString());

		wordWeights = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			String word = wordIdxer.getObject(w);
			double cnt = wordCnts.getCount(word);
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
			wordWeights.add(w, tfidf);
		}
	}

	public Counter<String> estimate(String outFileName) throws Exception {
		phrsIdxer = Generics.newIndexer(phrsCnts.keySet());

		computeWordWeights();

		SparseMatrix T = getSimilarityMatrix(getFeatureMatrix());

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = cents.copy();

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
			biases.add(i, weight);
		}

		biases.normalize();

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

		// System.out.println(ret.toStringSortedByValues(true, true, 100, "\t"));

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
		VectorMath.unitVector(F);
		return F;
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

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger row_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(X, row_cnt, ii, timer)));
		}

		CounterMap<Integer, Integer> ret = Generics.newCounterMap();

		for (int i = 0; i < thread_size; i++) {
			CounterMap<Integer, Integer> cm = fs.get(i).get();
			ret.incrementAll(cm);
		}
		fs.clear();
		tpe.shutdown();

		ListMap<Integer, Integer> lm = Generics.newListMap(ListType.LINKED_LIST);

		for (int i : ret.keySet()) {
			for (int j : ret.getCounter(i).keySet()) {
				lm.put(i, j);
			}
		}

		for (int i : lm.keySet()) {
			LinkedList<Integer> js = (LinkedList<Integer>) lm.get(i);
			Counter<Integer> c = ret.getCounter(i);
			for (int j : js) {
				if (i != j) {
					double v = c.getCount(j);
					ret.incrementCount(j, i, v);
				}
			}
			js.clear();
		}

		SparseMatrix T = new SparseMatrix(ret);
		T.normalizeColumns();

		return T;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
