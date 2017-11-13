package ohs.ir.search.app;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.search.model.WordProximities;
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
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class ConceptWeightEstimator {

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

		Counter<String> tmpCnts = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			double tfidf = Double.parseDouble(ps[1]);
			double cnt = Double.parseDouble(ps[2]);
			double doc_freq = Double.parseDouble(ps[3]);
			tmpCnts.setCount(phrs, cnt);
		}

		Counter<String> phrsCnts = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_filtered.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			double cnt = tmpCnts.getCount(phrs);
			if (cnt > 50) {
				phrsCnts.setCount(phrs, cnt);
			}
		}

		Counter<String> seedPhrsCnts = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_seed.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			double cnt = tmpCnts.getCount(phrs);
			if (cnt > 0) {
				seedPhrsCnts.setCount(phrs, cnt);
			}
		}

		Vocab vocab = DocumentCollection.readVocab(MIRPath.DATA_DIR + "merged/col/dc/vocab.ser");

		Map<String, String> wordToLemma = Generics.newHashMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/lemma.txt")) {
			String[] ps = line.split("\t");
			String word = ps[0];
			String lemma = ps[1];
			wordToLemma.put(word, lemma);
		}

		Set<String> stopwords = FileUtils.readStringHashSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(vocab, stopwords);

		Counter<String> clueWords = Generics.newCounter();

		for (String s : seedPhrsCnts.keySet()) {
			String[] ps = s.split("\t");
			String phrs = ps[0];

			for (String word : StrUtils.split(phrs)) {
				if (wf.filter(word)) {
					continue;
				}
				clueWords.incrementCount(word, 1);
			}
		}

		SetMap<String, String> abbrMap = FileUtils.readStringSetMapFromText(MIRPath.PHRS_DIR + "abbr_tok.txt");

		ConceptWeightEstimator pre = new ConceptWeightEstimator(vocab, wf, phrsCnts, seedPhrsCnts, clueWords,
				wordToLemma, abbrMap);
		pre.setThreadSize(10);
		pre.setMinCosine(0.2);
		pre.setMinFeatSize(10);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		System.out.println("process ends.");
	}

	private SetMap<String, String> abbrMap;

	private Counter<String> clueWords;

	private double min_cosine = 0.3;

	private int min_feat_size = 30;

	private Counter<String> phrsCnts;

	private Indexer<String> phrsIdxer;

	private Counter<String> seedPhrsCnts;

	private int thread_size = 5;

	private Vocab vocab;

	private WordFilter wf;

	private Map<Integer, Integer> wordToLemma;

	private DenseVector wordWeights;

	public ConceptWeightEstimator(Vocab vocab, WordFilter wf, Counter<String> phrsCnts, Counter<String> seedPhrsCnts,
			Counter<String> clueWords, Map<String, String> wordToLemma, SetMap<String, String> abbrMap) {
		this.vocab = vocab;
		this.wf = wf;
		this.phrsCnts = phrsCnts;
		this.seedPhrsCnts = seedPhrsCnts;

		this.clueWords = clueWords;
		this.wordToLemma = Generics.newHashMap();

		for (Entry<String, String> e : wordToLemma.entrySet()) {
			String word = e.getKey();
			String lemma = e.getValue();

			int w = vocab.indexOf(word);
			int l = vocab.indexOf(lemma);

			if (w < 0 || l < 0) {
				continue;
			}

			this.wordToLemma.put(w, l);
		}

		this.abbrMap = abbrMap;

		phrsIdxer = Generics.newIndexer(phrsCnts.keySet());
	}

	private void computeWordWeights() {
		wordWeights = new DenseVector(vocab.size());
		for (int w = 0; w < vocab.size(); w++) {
			double cnt = vocab.getCount(w);
			double doc_cnt = vocab.getDocCnt();
			double doc_freq = vocab.getDocFreq(w);

			if (doc_freq < 100) {
				continue;
			}

			double tfidf = TermWeighting.tfidf(cnt, doc_cnt, doc_freq);
			wordWeights.add(w, tfidf);
		}
	}

	public Counter<String> estimate(String outFileName) throws Exception {

		computeWordWeights();

		// SparseMatrix T =
		// getSimilarityMatrix(getFeatureMatrix(getCooccurrenceMatrix()));
		SparseMatrix T = getSimilarityMatrix(getFeatureMatrix(null));

		DenseVector P = new DenseVector(phrsIdxer.size());
		DenseVector B = getBiases();

		System.out.println(VectorUtils.toCounter(B, phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		System.out.println("run random-walk");

		VectorMath.randomWalk(T, P, B, 500, 0.000001, 0.85, thread_size);

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

	private DenseVector getBiases() {
		System.out.println("get phrase biases");

		System.out.println(clueWords.toString());

		DenseVector ret = new DenseVector(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);

			List<String> words = StrUtils.split(phrs);
			double weight = 0;

			for (int j = 0; j < words.size(); j++) {
				String word = words.get(j);
				double cnt = clueWords.getCount(word);
				int w = vocab.indexOf(word);

				if (!clueWords.containsKey(word) || w < 0) {
					continue;
				}

				double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
				weight += tfidf;
			}

			weight /= words.size();
			ret.add(i, weight);
		}

		ret.normalize();
		return ret;
	}

	private DenseVector getBiases2() {
		System.out.println("get phrase biases");

		DenseVector ret = new DenseVector(phrsIdxer.size());
		ret.add(1);

		CounterMap<String, String> bigrams = Generics.newCounterMap();
		Counter<String> unigrams = Generics.newCounter();

		for (String phrs : phrsCnts.keySet()) {
			double cnt = phrsCnts.getCount(phrs);

			List<String> words = StrUtils.split(phrs);

			for (int i = 0; i < words.size(); i++) {
				String word1 = words.get(i);

				if (i < words.size() - 1) {
					String word2 = words.get(i + 1);
					bigrams.incrementCount(word1, word2, 1);
				}
				unigrams.incrementCount(word1, 1);
			}
		}

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			double p_weight = 0;

			List<String> words = StrUtils.split(phrs);

			for (int j = 0; j < words.size(); j++) {
				String word1 = words.get(j);
				int w = vocab.indexOf(word1);
				double weight = wordWeights.value(w);
				double cnt1 = unigrams.getCount(word1);
				double cnt2 = 0;

				if (j < words.size() - 1) {
					String word2 = words.get(j + 1);
					cnt2 = bigrams.getCount(word1, word2);
				}

				if (cnt1 > 0) {
					cnt1 = 1;
				}

				if (cnt2 > 0) {
					cnt2 = 2;
				}

				double cnt3 = cnt1 + cnt2;

				if (cnt1 > 0) {
					// p_weight += weight;
					p_weight += cnt1;
				}
			}

			p_weight /= words.size();

			ret.add(i, p_weight);
		}

		ret.normalize();

		return ret;
	}

	private DenseVector getBiases3() {
		System.out.println("get phrase biases");

		CounterMap<String, String> bigrams = Generics.newCounterMap();
		Counter<String> unigrams = Generics.newCounter();

		for (String phrs : phrsCnts.keySet()) {
			double cnt = phrsCnts.getCount(phrs);

			List<String> words = StrUtils.split(phrs);

			for (int i = 0; i < words.size(); i++) {
				String word1 = words.get(i);

				if (i < words.size() - 1) {
					String word2 = words.get(i + 1);
					bigrams.incrementCount(word1, word2, cnt);
				}
				unigrams.incrementCount(word1, cnt);
			}
		}

		double mixture_b = 0.5;

		Counter<Integer> c = Generics.newCounter();

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			double pr_sum = 0;

			List<String> words = StrUtils.split(phrs);

			for (int j = 0; j < words.size(); j++) {
				String word1 = words.get(j);
				double pr_u = unigrams.getProbability(word1);
				double pr_b = 0;

				if (j < words.size() - 1) {
					String word2 = words.get(j + 1);
					pr_b = bigrams.getProbability(word1, word2);
				}

				double pr = (1 - mixture_b) * pr_u + mixture_b * pr_b;

				if (pr > 0) {
					pr_sum += Math.log(pr);
				}
			}

			if (pr_sum != 0) {
				c.setCount(i, pr_sum);
			}
		}

		SparseVector prs = new SparseVector(c);
		VectorMath.softmax(prs);

		DenseVector ret = new DenseVector(phrsIdxer.size());

		for (int i = 0; i < prs.size(); i++) {
			int j = prs.indexAt(i);
			double v = prs.valueAt(i);
			ret.add(j, v);
		}
		ret.normalize();
		return ret;
	}

	private SparseMatrix getCooccurrenceMatrix() throws Exception {
		System.out.println("get cooccurrence matrix.");

		File file = new File(MIRPath.PHRS_DIR + "phrs_word_cooccur.txt");

		CounterMap<Integer, Integer> cm = Generics.newCounterMap(phrsIdxer.size());

		// if (file.exists()) {
		// TextFileReader reader = new TextFileReader(file);
		// while (reader.hasNext()) {
		// String[] ps = reader.next().split("\t");
		// int w1 = vocab.indexOf(ps[0]);
		// int w2 = vocab.indexOf(ps[1]);
		// double v = Double.parseDouble(ps[2]);
		// cm.incrementCount(w1, w2, v);
		// }
		// reader.close();
		// } else {
		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);
			double cnt = phrsCnts.getCount(phrs);

			List<String> words = StrUtils.split(phrs);
			IntegerArray ws = new IntegerArray(vocab.indexesOf(words, -1));
			cm.incrementAll(WordProximities.hal(ws, ws.size(), false), cnt);
		}

		WordProximities.symmetric(cm);

		Set<Integer> clueSet = Generics.newHashSet(vocab.indexesOfKnown(clueWords.keySet()));

		for (Entry<Integer, Counter<Integer>> e : cm.getEntrySet()) {
			e.getValue().pruneExcept(clueSet);
		}

		weightTFIDFs(cm);

		TextFileWriter writer = new TextFileWriter(file);

		List<String> words1 = vocab.getObjects(cm.keySet());
		Collections.sort(words1);

		for (String word1 : words1) {
			int w1 = vocab.indexOf(word1);
			Counter<Integer> c = cm.getCounter(w1);

			List<String> words2 = vocab.getObjects(c.keySet());
			Collections.sort(words2);

			for (String word2 : words2) {
				int w2 = vocab.indexOf(word2);
				double v = c.getCount(w2);
				if (v > 0) {
					writer.write(String.format("%s\t%s\t%f\n", word1, word2, v));
				}
			}
		}
		writer.close();
		// }

		System.out.println(VectorUtils.toCounterMap(cm, vocab, vocab));

		SparseMatrix C = new SparseMatrix(cm);
		C.normalizeColumns();
		return C;
	}

	private SparseMatrix getFeatureMatrix(SparseMatrix C) {
		System.out.println("get feature matrix.");

		CounterMap<Integer, Integer> cm = Generics.newCounterMap(phrsIdxer.size());

		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);

			for (String word : StrUtils.split(phrs)) {
				int w = vocab.indexOf(word);
				if (wf.filter(w)) {
					continue;
				}

				cm.incrementCount(p, w, 1);

				if (wordToLemma != null) {
					Integer l = wordToLemma.get(w);
					if (l != null && w != l) {
						cm.incrementCount(p, l, 1);
					}
				}
			}

			Set<String> longForms = abbrMap.get(phrs, false);

			if (longForms != null && longForms.size() == 1) {
				List<String> ll = Generics.newArrayList(longForms);
				for (String word : StrUtils.split(ll.get(0))) {
					int w = vocab.indexOf(word);
					if (wf.filter(w)) {
						continue;
					}

					cm.incrementCount(p, w, 1);

					if (wordToLemma != null) {
						Integer l = wordToLemma.get(w);
						if (l != null && w != l) {
							cm.incrementCount(p, l, 1);
						}
					}
				}
			}
		}

		{
			CounterMap<Integer, Integer> cm2 = Generics.newCounterMap();

			for (int p : cm.keySet()) {
				Counter<Integer> c1 = cm.getCounter(p);
				Counter<Integer> c2 = Generics.newCounter();
				for (Entry<Integer, Double> e : c1.entrySet()) {
					int w = e.getKey();
					double weight = wordWeights.value(w);

					if (weight == 0) {
						continue;
					}

					c2.setCount(w, weight);
				}
				cm2.setCounter(p, c2);
			}
			cm = cm2;
		}

		if (C != null) {
			for (int p : cm.keySet()) {
				Counter<Integer> c1 = cm.getCounter(p);
				Counter<Integer> c2 = Generics.newCounter();

				for (int w1 : c1.keySet()) {
					double weight1 = wordWeights.value(w1);
					c2.incrementCount(w1, weight1);

					SparseVector v2 = C.row(w1);
					VectorMath.addAfterMultiply(v2, weight1, c2);
				}

				c2.keepTopNKeys(min_feat_size);
				cm.setCounter(p, c2);
			}
		}

		return new SparseMatrix(cm);
	}

	private SparseMatrix getSimilarityMatrix(SparseMatrix X) throws Exception {
		System.out.println("get similarity matrix");

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

		CounterMap<Integer, Integer> cm = Generics.newCounterMap();

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(X, ii, min_cosine, row_cnt, cm, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		tpe.shutdown();

		WordProximities.symmetric(cm);

		SparseMatrix T = new SparseMatrix(cm);
		T.normalizeColumns();

		return T;
	}

	public void setMinCosine(double min_cosine) {
		this.min_cosine = min_cosine;
	}

	public void setMinFeatSize(int min_feat_size) {
		this.min_feat_size = min_feat_size;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	private void weightTFIDFs(CounterMap<Integer, Integer> cm) {
		for (int w1 : cm.keySet()) {
			double tfidf1 = wordWeights.value(w1);
			Counter<Integer> c = cm.getCounter(w1);
			for (Entry<Integer, Double> e : c.entrySet()) {
				int w2 = e.getKey();
				double sim = e.getValue();
				double tfidf2 = wordWeights.value(w2);
				cm.setCount(w1, w2, sim * tfidf1 * tfidf2);
			}
		}
	}

}
