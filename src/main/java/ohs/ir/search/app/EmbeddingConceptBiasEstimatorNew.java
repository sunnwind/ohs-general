package ohs.ir.search.app;

import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import com.sun.xml.bind.v2.runtime.unmarshaller.StructureLoader;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
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
public class EmbeddingConceptBiasEstimatorNew {

	class Worker1 implements Callable<CounterMap<Integer, Integer>> {

		private AtomicInteger row_cnt;

		private Timer timer;

		public Worker1(AtomicInteger range_cnt, Timer timer) {
			this.row_cnt = range_cnt;
			this.timer = timer;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {
			int m = 0;

			while ((m = row_cnt.getAndIncrement()) < X.rowSize()) {
				DenseVector x = X.rowAt(m);
				double cosine = VectorMath.cosine(x, Y);
				phrsBiases.add(m, cosine);
			}

			int prog = BatchUtils.progress(m + 1, X.rowSize());

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d, %s]\n", prog, m + 1, X.rowSize(), timer.stop());
			}

			return null;
		}
	}

	public static DenseMatrix getFeatureMatrix1(RandomAccessDenseMatrix E, Indexer<String> idxer, Vocab vocab,
			WordFilter wf) throws Exception {
		DenseMatrix X = new DenseMatrix(idxer.size(), E.colSize());

		for (int p = 0; p < idxer.size(); p++) {
			String phrs = idxer.getObject(p);
			DenseVector x = X.row(p);
			int cnt = 0;

			List<String> words = StrUtils.split(phrs);
			double idf_sum = 0;

			for (String word : words) {
				int w = vocab.indexOf(word);

				if (wf.filter(w)) {
					continue;
				}

				double idf = TermWeighting.tfidf(vocab.getCount(w), vocab.getDocCnt(), vocab.getDocFreq(w));
				idf /= idxer.size();
				idf_sum += idf;

				DenseVector e = E.row(w);
				VectorMath.add(e, x);
				// VectorMath.addAfterMultiply(e, idf, x);
				cnt++;
			}

			if (cnt > 0) {
				x.multiply(1d / cnt);
				// x.multiply(1d / idf_sum);
			}

			double ratio = 1d * cnt / words.size();

			if (ratio <= 0.5) {
				x.setAll(0);
			}

			int prog = BatchUtils.progress(p + 1, idxer.size());

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d]\n", prog, p + 1, idxer.size());
			}
		}

		// X.multiply(1f / idf_sum);

		return X;
	}

	public static DenseMatrix getFeatureMatrix2(RandomAccessDenseMatrix E, Indexer<String> idxer, Vocab vocab,
			WordFilter wf) throws Exception {
		DenseMatrix X = new DenseMatrix(idxer.size(), E.colSize() * 2);

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int p = 0; p < idxer.size(); p++) {
			String phrs = idxer.getObject(p);

			List<String> words = StrUtils.split(phrs);

			for (int i = 0; i < words.size(); i++) {
				String word1 = words.get(i);
				int w1 = vocab.indexOf(word1);
				if (wf.filter(w1)) {
					continue;
				}

				if (i < words.size() - 1) {
					String word2 = words.get(i + 1);
					int w2 = vocab.indexOf(word2);
					if (wf.filter(w2)) {
						continue;
					}
					cm.incrementCount(word1, word2, 1);
				}
			}
		}

		for (int p = 0; p < idxer.size(); p++) {
			String phrs = idxer.getObject(p);
			DenseVector x = X.row(p);
			int cnt = 0;
			List<String> words = StrUtils.split(phrs);

			for (String word : words) {
				int w = vocab.indexOf(word);

				if (wf.filter(w)) {
					continue;
				}

				DenseVector e = E.row(w);
				VectorMath.add(e, x);
				// VectorMath.addAfterMultiply(e, idf, x);
				cnt++;
			}

			if (cnt > 0) {
				x.multiply(1d / cnt);
				// x.multiply(1d / idf_sum);
			}

			double ratio = 1d * cnt / words.size();

			if (ratio <= 0.5) {
				x.setAll(0);
			}

			int prog = BatchUtils.progress(p + 1, idxer.size());

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d]\n", prog, p + 1, idxer.size());
			}
		}

		return X;
	}

	public static DenseVector getFeatureVector(RandomAccessDenseMatrix E, Indexer<String> idxer, Vocab vocab,
			WordFilter wf) throws Exception {
		DenseVector x = new DenseVector(E.colSize());
		int cnt = 0;
		double idf_sum = 0;

		for (int p = 0; p < idxer.size(); p++) {
			String phrs = idxer.getObject(p);
			List<String> words = StrUtils.split(phrs);

			for (String word : words) {
				int w = vocab.indexOf(word);

				if (wf.filter(w)) {
					continue;
				}

				double idf = TermWeighting.tfidf(vocab.getCount(w), vocab.getDocCnt(), vocab.getDocFreq(w));
				idf /= idxer.size();
				idf_sum += idf;

				DenseVector e = E.row(w);
				VectorMath.add(e, x);
				// VectorMath.addAfterMultiply(e, idf, x);
				cnt++;
			}
		}
		x.multiply(1d / cnt);
		return x;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsDocFreqs = Generics.newCounter();
		Counter<String> phrsTfidfs = Generics.newCounter();

		List<String> seeds = Generics.newArrayList();
		{
			List<String> lines = FileUtils.readLinesFromText(MIRPath.PHRS_DIR + "phrs_medical_data.txt");

			for (int i = 2; i < lines.size(); i++) {
				List<String> ps = StrUtils.split("\t", lines.get(i));
				ps = StrUtils.unwrap(ps);
				String type = ps.get(0);
				if (!type.equals("Symptom & Sign")) {
					continue;
				}
				String phrs = ps.get(1).trim();
				seeds.add(phrs);
			}

			EnglishTokenizer et = new EnglishTokenizer();

			for (int i = 0; i < seeds.size(); i++) {
				String phrs = seeds.get(i);
				phrs = StrUtils.join(" ", et.tokenize(phrs));
				seeds.set(i, phrs);
			}
		}

		System.out.println();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			double tfidf = Double.parseDouble(ps[1]);
			double cnt = Double.parseDouble(ps[2]);
			double doc_freq = Double.parseDouble(ps[3]);
			phrsDocFreqs.setCount(phrs, cnt);
			phrsTfidfs.setCount(phrs, tfidf);
		}

		Counter<String> seedPhrsCnts = Generics.newCounter();

		// for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR +
		// "phrs/phrs_seed.txt")) {
		// String[] ps = line.split("\t");
		// String phrs = ps[0];
		// double rsc_size = Double.parseDouble(ps[1]);
		// double cnt = phrsDocFreqs.getCount(phrs);
		// if (cnt > 0) {
		// seedPhrsCnts.setCount(phrs, cnt);
		// }
		// }

		seedPhrsCnts.incrementAll(seeds, 1);

		// for (String phrs : seeds) {
		// double cnt = phrsDocFreqs.getCount(phrs);
		//
		// // if (cnt > 0) {
		// seedPhrsCnts.setCount(phrs, 1);
		// // }else {
		// // System.out.println(phrs);
		// // }
		// }

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsDocFreqs.keySet());
		Indexer<String> seedPhrsIdxer = Generics.newIndexer(seedPhrsCnts.keySet());

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser");

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(dc.getVocab(), stopwords);
		Vocab vocab = dc.getVocab();

		DenseMatrix X = getFeatureMatrix1(E, phrsIdxer, vocab, wf);
		DenseVector Y = getFeatureVector(E, seedPhrsIdxer, vocab, wf);

		// getFeatureMatrix2(E, seedPhrsIdxer, vocab, wf);

		// DenseVector pWeights = VectorUtils.toSparseVector(phrsDocFreqs,
		// phrsIdxer).toDenseVector(phrsIdxer.size());

		DenseVector pWeights = new DenseVector(phrsIdxer.size());

		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);
			double weight = 0;
			double size = 0;

			for (String word : StrUtils.split(phrs)) {
				if (wf.filter(word)) {
					continue;
				}

				double cnt = vocab.getCount(word);
				double doc_freq = vocab.getDocFreq(word);

				if (cnt == 0 || doc_freq == 0) {
					continue;
				}
				weight += TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(word));
				size++;
			}

			if (size > 0) {
				weight /= size;
				pWeights.add(p, weight);
			}
		}

		EmbeddingConceptBiasEstimatorNew pre = new EmbeddingConceptBiasEstimatorNew(phrsIdxer, X, seedPhrsIdxer, Y, pWeights);

		pre.setThreadSize(8);
		pre.setTopK(Integer.MAX_VALUE);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_bias.txt");

		System.out.println("process ends.");
	}

	private DenseVector phrsBiases;

	private Indexer<String> phrsIdxer;

	private DenseVector phrsTfidfs;

	private Indexer<String> seedIdxer;

	private int thread_size = 5;

	private int top_k = 10;

	private DenseMatrix X;

	private DenseVector Y;

	public EmbeddingConceptBiasEstimatorNew(Indexer<String> phrsIdxer, DenseMatrix X, Indexer<String> seedIdxer,
			DenseVector Y, DenseVector phrsTfidfs) {
		this.phrsIdxer = phrsIdxer;
		this.X = X;
		this.seedIdxer = seedIdxer;
		this.Y = Y;
		this.phrsTfidfs = phrsTfidfs;
	}

	public void estimate(String outFileName) throws Exception {
		Timer timer = Timer.newTimer();

		phrsBiases = new DenseVector(phrsIdxer.size());

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger row_cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker1(row_cnt, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		tpe.shutdown();

		FileUtils.writeStringCounterAsText(outFileName, VectorUtils.toCounter(phrsBiases, phrsIdxer));
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	public void setTopK(int top_k) {
		this.top_k = top_k;
	}
}
