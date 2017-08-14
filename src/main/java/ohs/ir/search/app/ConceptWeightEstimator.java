package ohs.ir.search.app;

import java.io.File;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

import org.apache.commons.math.stat.descriptive.SynchronizedMultivariateSummaryStatistics;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.medical.query.TrecCdsQuery;
import ohs.ir.search.index.WordFilter;
import ohs.ir.search.model.WordProximities;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
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
import ohs.types.number.IntegerArray;
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

			{
				List<String> lines = FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt");
				docFreqs = Generics.newCounter(lines.size());

				for (String line : lines) {
					String[] ps = line.split("\t");
					String phrs = ps[0];
					double tfidf = Double.parseDouble(ps[1]);
					double cnt = Double.parseDouble(ps[2]);
					double doc_freq = Double.parseDouble(ps[3]);
					docFreqs.setCount(phrs, doc_freq);
					// tfidfs.setCount(phrs, tfidf);
				}
			}

			int size_old = docFreqs.size();
			docFreqs.pruneKeysBelowThreshold(1000);
			phrsCnts = docFreqs;
			System.out.printf("size: %d->%d\n", size_old, phrsCnts.size());
		}

		Vocab vocab = DocumentCollection.readVocab(MIRPath.DATA_DIR + "merged/col/dc/vocab.ser");

		Map<Integer, Integer> wordToLemma = Generics.newHashMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/lemma.txt")) {
			String[] ps = line.split("\t");
			String word = ps[0];
			String lemma = ps[1];

			int w = vocab.indexOf(word);
			int l = vocab.indexOf(lemma);

			if (w < 0 || l < 0) {
				continue;
			}
			wordToLemma.put(w, l);
		}

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(vocab, stopwords);

		Counter<String> clueWords = Generics.newCounter();
		Counter<String> cluePhrsCnts = Generics.newCounter();

		for (String s : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_filtered.txt")) {
			String[] ps = s.split("\t");
			String phrs = ps[0];
			int rsc_cnt = Integer.parseInt(ps[1]);
			cluePhrsCnts.setCount(phrs, rsc_cnt);

			for (String word : StrUtils.split(phrs)) {
				clueWords.incrementCount(word, 1);
			}
		}

		// {
		// StringTokenizer st = new EnglishTokenizer();
		//
		// Pattern puncPat = Pattern.compile("^\\p{Punct}+");
		//
		// for (BaseQuery bq : QueryReader.readQueries(MIRPath.TREC_PM_2017_QUERY_FILE))
		// {
		// List<String> words = st.tokenize(bq.getSearchText());
		//
		// for (String word : words) {
		// if (wf.filter(word)) {
		// continue;
		// }
		// clueWords.incrementCount(word, 1);
		// }
		// }
		//
		// for (BaseQuery bq :
		// QueryReader.readQueries(MIRPath.TREC_CDS_2014_QUERY_FILE)) {
		// List<String> words = st.tokenize(bq.getSearchText());
		//
		// for (String word : words) {
		// if (wf.filter(word)) {
		// continue;
		// }
		// clueWords.incrementCount(word, 1);
		// }
		// }
		//
		// for (BaseQuery bq :
		// QueryReader.readQueries(MIRPath.TREC_CDS_2015_QUERY_A_FILE)) {
		// List<String> words = st.tokenize(bq.getSearchText());
		//
		// for (String word : words) {
		// if (wf.filter(word)) {
		// continue;
		// }
		// clueWords.incrementCount(word, 1);
		// }
		// }
		//
		// for (BaseQuery bq :
		// QueryReader.readQueries(MIRPath.TREC_CDS_2016_QUERY_FILE)) {
		// TrecCdsQuery tcq = (TrecCdsQuery) bq;
		// List<String> l = Generics.newArrayList();
		// l.add(tcq.getDescription());
		// l.add(tcq.getSummary());
		// l.add(tcq.getNote());
		//
		// String s = StrUtils.join("\n", l);
		//
		// List<String> words = st.tokenize(s);
		//
		// for (String word : words) {
		// if (wf.filter(word)) {
		// continue;
		// }
		// clueWords.incrementCount(word, 1);
		// }
		// }
		//
		// for (BaseQuery bq : QueryReader.readQueries(MIRPath.CLEF_EH_2016_QUERY_FILE))
		// {
		// List<String> words = st.tokenize(bq.getSearchText());
		//
		// for (String word : words) {
		// if (wf.filter(word)) {
		// continue;
		// }
		// clueWords.incrementCount(word, 1);
		// }
		// }
		//
		// for (String word : Generics.newArrayList(clueWords.keySet())) {
		// int w = vocab.indexOf(word);
		// if (w < 0) {
		// continue;
		// }
		// Integer l = wordToLemma.get(w);
		//
		// if (l == null) {
		// continue;
		// }
		//
		// String lemma = vocab.getObject(l);
		//
		// clueWords.incrementCount(lemma, 1);
		// }
		// }

		SetMap<String, String> abbrMap = FileUtils.readStringSetMapFromText(MIRPath.PHRS_DIR + "abbr_tok.txt");

		ConceptWeightEstimator pre = new ConceptWeightEstimator(vocab, wf, phrsCnts, clueWords, wordToLemma, abbrMap);
		pre.setThreadSize(10);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

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

	public ConceptWeightEstimator(Vocab vocab, WordFilter wf, Counter<String> phrsCnts, Counter<String> clueWords,
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
			double cnt = vocab.getCount(word);
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(word));
			wordWeights.add(w, tfidf);
		}
	}

	public Counter<String> estimate(String outFileName) throws Exception {

		computeWordWeights();

		SparseMatrix T = getSimilarityMatrix(getFeatureMatrix(getCooccurrenceMatrix()));

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = getPhraseBiases();

		VectorMath.randomWalk(T, cents, biases, 500, 0.000001, 0.85, thread_size);

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

	private SparseMatrix getCooccurrenceMatrix() {
		System.out.println("get cooccurrence matrix.");

		CounterMap<Integer, Integer> cm = Generics.newCounterMap(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			int p = phrsIdxer.indexOf(phrs);
			if (p < 0) {
				continue;
			}

			List<String> words = StrUtils.split(phrs);

			IntegerArray ws = new IntegerArray(words.size());

			for (String word : words) {
				int w = wordIdxer.indexOf(word);
				ws.add(w);
			}

			cm.incrementAll(WordProximities.hal(ws, ws.size(), false));
		}

		cm = WordProximities.symmetric(cm);

		{
			Set<Integer> clueSet = Generics.newHashSet(wordIdxer.indexesOfKnown(clueWords.keySet()));

			for (Entry<Integer, Counter<Integer>> e : cm.getEntrySet()) {
				e.getValue().pruneExcept(clueSet);
			}
		}

		weightTFIDFs(cm);

		System.out.println(VectorUtils.toCounterMap(cm, wordIdxer, wordIdxer));

		SparseMatrix C = new SparseMatrix(cm);
		C.normalizeColumns();
		return C;
	}

	private SparseMatrix getFeatureMatrix(SparseMatrix C) {
		System.out.println("get feature matrix.");

		CounterMap<Integer, Integer> cm = Generics.newCounterMap(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			int p = phrsIdxer.indexOf(phrs);
			if (p < 0) {
				continue;
			}

			for (String word : StrUtils.split(phrs)) {
				int w = wordIdxer.indexOf(word);
				if (w < 0) {
					continue;
				}

				cm.incrementCount(p, w, 1);

				// if (wordToLemma != null) {
				// Integer l = wordToLemma.get(w);
				// if (l != null) {
				// cm.incrementCount(p, l, 1);
				// }
				// }
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

		for (int i = 0; i < F.rowSize(); i++) {
			int p = F.indexAt(i);
			SparseVector pv = F.rowAt(i);
			Counter<Integer> c = Generics.newCounter();

			for (int j = 0; j < pv.size(); j++) {
				int w1 = pv.indexAt(j);
				double weight1 = wordWeights.value(w1);
				SparseVector wv = C.row(w1);

				for (int k = 0; k < wv.size(); k++) {
					int w2 = wv.indexAt(k);
					double weight2 = wv.valueAt(k);
					c.incrementCount(w2, weight1 * weight2);
				}
			}

			c.keepTopNKeys(100);

			F.setRowAt(i, new SparseVector(c));
		}

		return F;
	}

	private DenseVector getPhraseBiases() {
		System.out.println("get phrase biases");

		DenseVector ret = new DenseVector(phrsIdxer.size());

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			double weight = 0;
			double cnt2 = 0;
			for (String word : StrUtils.split(phrs)) {
				// weight += clueWords.getCount(word);
				int w = wordIdxer.indexOf(word);
				if (w < 0) {
					continue;
				}
				double cnt = clueWords.getCount(word);

				if (cnt > 0) {
					weight += wordWeights.value(w);
					cnt2++;
				}
			}

			double rsc_cnt = phrsCnts.getCount(phrs);

			ret.add(i, cnt2);
		}

		ret.normalize();

		return ret;
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

	private void weightTFIDFs(CounterMap<Integer, Integer> cm) {
		for (int w1 : cm.keySet()) {
			Counter<Integer> c = cm.getCounter(w1);
			for (Entry<Integer, Double> e : c.entrySet()) {
				int w2 = e.getKey();
				double sim = e.getValue();
				double tfidf = wordWeights.value(w2);
				cm.setCount(w1, w2, sim * tfidf);
			}
		}
	}

}
