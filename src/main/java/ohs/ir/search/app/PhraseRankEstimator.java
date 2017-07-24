package ohs.ir.search.app;

import java.io.File;
import java.util.List;
import java.util.Set;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.query.BaseQuery;
import ohs.ir.medical.query.QueryReader;
import ohs.ir.search.app.PhraseWeightEstimator.Worker;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListList;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;
import xtc.tree.GNode;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class PhraseRankEstimator {

	class Worker implements Callable<CounterMap<Integer, Integer>> {

		private AtomicInteger row_cnt;

		private SparseMatrix T;

		private Timer timer;

		public Worker(SparseMatrix T, AtomicInteger range_cnt, Timer timer) {
			this.T = T;
			this.row_cnt = range_cnt;
			this.timer = timer;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {
			int i = 0;

			CounterMap<Integer, Integer> cm = Generics.newCounterMap();

			while ((i = row_cnt.getAndIncrement()) < T.rowSize()) {
				int p1 = T.indexAt(i);
				SparseVector t1 = T.rowAt(i);

				for (int j = i + 1; j < T.rowSize(); j++) {
					int p2 = T.indexAt(j);
					SparseVector t2 = T.rowAt(j);

					double cosine = VectorMath.cosine(t1, t2);
					cosine = Math.max(0, cosine);

					if (cosine > 0) {
						cm.setCount(p1, p2, cosine);
					}
				}

				int prog = BatchUtils.progress(i, T.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, i, T.rowSize(), timer.stop());
				}
			}

			return cm;
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
				phrsCnts.setCount(ps[0], Double.parseDouble(ps[2]));
			}
		}

		phrsCnts.pruneKeysBelowThreshold(5000);

		StringTokenizer st = new EnglishTokenizer();

		List<BaseQuery> bqs = QueryReader.readQueries(MIRPath.TREC_PM_2017_QUERY_FILE);

		Counter<String> clueWords = Generics.newCounter();

		for (BaseQuery bq : bqs) {
			List<String> words = st.tokenize(bq.getSearchText());

			for (String word : words) {
				clueWords.incrementCount(word, 1);
			}
		}

		// DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR +
		// "merged/col/dc/");

		Vocab vocab = DocumentCollection.readVocab(MIRPath.DATA_DIR + "merged/col/dc/vocab.ser");

		PhraseRankEstimator pre = new PhraseRankEstimator(vocab, null, phrsCnts, clueWords);
		pre.setThreadSize(10);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_weight.txt");

		System.out.println("process ends.");
	}

	private int thread_size = 5;

	private Vocab vocab;

	private WordFilter wf;

	private Counter<String> phrsCnts;

	private Indexer<String> phrsIdxer;

	private Indexer<String> wordIdxer;

	private Counter<String> clueWords;

	public PhraseRankEstimator(Vocab vocab, WordFilter wf, Counter<String> phrsCnts, Counter<String> clueWords) {
		this.vocab = vocab;
		this.wf = wf;
		this.phrsCnts = phrsCnts;
		this.clueWords = clueWords;
	}

	private DenseVector getWordWeights(Indexer<String> wordIdxer, Counter<String> wordCnts) {
		DenseVector ret = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			String t = wordIdxer.getObject(w);
			double doc_freq = vocab.getDocFreq(t);
			double cnt = wordCnts.getCount(t);
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);
			ret.add(w, tfidf);
		}
		return ret;
	}

	private DenseVector getPhraseWeights(Indexer<String> phrsIdxer, Indexer<String> wordIdxer,
			DenseVector wordWeights) {
		DenseVector ret = new DenseVector(phrsIdxer.size());
		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);
			double weight = 0;
			for (String word : StrUtils.split(phrs)) {
				int w = wordIdxer.indexOf(word);
				weight += wordWeights.value(w);
			}
			ret.add(p, weight);
		}
		return ret;
	}

	private SparseMatrix getSimMatrix(SparseMatrix T) throws Exception {
		Timer timer = Timer.newTimer();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger row_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(T, row_cnt, timer)));
		}

		CounterMap<Integer, Integer> ret = Generics.newCounterMap();

		for (int i = 0; i < thread_size; i++) {
			CounterMap<Integer, Integer> cm = fs.get(i).get();

			for (Entry<Integer, Counter<Integer>> e1 : cm.getEntrySet()) {
				int p1 = e1.getKey();
				Counter<Integer> c = e1.getValue();

				for (Entry<Integer, Double> e2 : c.entrySet()) {
					int p2 = e2.getKey();
					double val = e2.getValue();
					ret.setCount(p1, p2, val);
					ret.setCount(p2, p1, val);
				}
			}
			cm = null;
		}
		fs.clear();

		tpe.shutdown();

		return new SparseMatrix(ret);
	}

	public Counter<String> estimate(String outFileName) throws Exception {
		phrsIdxer = Generics.newIndexer(phrsCnts.keySet());

		Counter<String> wordCnts = Generics.newCounter();

		for (String phrs : phrsIdxer.getObjects()) {
			for (String word : StrUtils.split(phrs)) {
				wordCnts.incrementCount(word, 1);
			}
		}

		Indexer<String> wordIdxer = Generics.newIndexer(wordCnts.keySet());

		DenseVector wordWeights = getWordWeights(wordIdxer, wordCnts);
		DenseVector phrsWeights = getPhraseWeights(phrsIdxer, wordIdxer, wordWeights);

		// System.out.println(cm1.toString());

		SparseMatrix T = null;

		{
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
				double p_weight = phrsWeights.value(p);
				Counter<Integer> c = cm.getCounter(p);

				for (Entry<Integer, Double> e : c.entrySet()) {
					int w = e.getKey();
					double w_weight = wordWeights.value(w);
					cm.setCount(p, w, w_weight);
				}
			}

			T = new SparseMatrix(cm);

			cm = null;

			T = getSimMatrix(T);
		}

		T.normalizeColumns();

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = cents.copy();
		DenseVector biases2 = cents.copy();

		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			double cnt = 0;
			for (String word : StrUtils.split(phrs)) {
				cnt += clueWords.getCount(word);
			}
			biases.add(i, cnt);
		}

		biases.normalize();

		VectorMath.randomWalk(T, cents, biases, 500, 0.000001, 0.85);

		Counter<String> ret = Generics.newCounter();

		for (int w = 0; w < phrsIdxer.size(); w++) {
			String phrs = phrsIdxer.getObject(w);
			double c = cents.value(w);
			if (c > 0) {
				ret.incrementCount(phrs, c);
			}
		}

		FileUtils.writeStringCounterAsText(outFileName, ret);

		// System.out.println(ret.toStringSortedByValues(true, true, 100, "\t"));

		return ret;
	}

	private CounterMap<String, String> getWordToWords() {
		CounterMap<String, String> ret = Generics.newCounterMap();
		for (int i = 0; i < phrsIdxer.size(); i++) {
			String phrs = phrsIdxer.getObject(i);
			List<String> words = StrUtils.split(phrs);

			for (int j = 0; j < words.size(); j++) {
				String t1 = words.get(j);

				for (int k = j + 1; k < words.size(); k++) {
					String t2 = words.get(k);
					double dist = k - j;
					double score2 = 1d / dist;
					ret.incrementCount(t1, t2, score2);
				}
			}
		}
		return ret;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
