package ohs.ir.search.app;

import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
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
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class EmbeddingConceptBiasEstimatorOld {

	class Worker implements Callable<CounterMap<Integer, Integer>> {

		private AtomicInteger row_cnt;

		private Timer timer;

		public Worker(AtomicInteger range_cnt, Timer timer) {
			this.row_cnt = range_cnt;
			this.timer = timer;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {
			int m = 0;

			while ((m = row_cnt.getAndIncrement()) < X.rowSize()) {
				DenseVector x = X.rowAt(m);

				if (x.sum() == 0) {
					continue;
				}

				Counter<Integer> c = Generics.newCounter();

				for (int n = 0; n < Y.rowSize(); n++) {
					DenseVector y = Y.row(n);
					if (y.sum() == 0) {
						continue;
					}
					double p_tfidf = phrsTfidfs.value(n);
					c.incrementCount(n, VectorMath.cosine(x, y) * p_tfidf);
				}

				if (c.size() > 0) {

					if (top_k < Integer.MAX_VALUE) {
						c.keepTopNKeys(top_k);
					}

					synchronized (phrsBiases) {
						double bias = c.average();
						// for (Entry<Integer, Double> e : c.entrySet()) {
						// int p = e.getKey();
						// double v = e.getValue();
						// double w = phrsWeights.value(p);
						// bias += (v * w);
						// }
						// bias /= c.size();
						phrsBiases.add(m, c.totalCount());
					}
				}

				int prog = BatchUtils.progress(m + 1, X.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, m + 1, X.rowSize(), timer.stop());
				}
			}

			return null;
		}
	}

	public static DenseMatrix getFeatureMatrix(RandomAccessDenseMatrix E, Indexer<String> idxer, Vocab vocab,
			WordFilter wf) throws Exception {
		DenseMatrix X = new DenseMatrix(idxer.size(), E.colSize());

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

				double idf = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w));

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

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsDocFreqs = Generics.newCounter();
		Counter<String> phrsTfidfs = Generics.newCounter();

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

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_seed.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			double rsc_size = Double.parseDouble(ps[1]);
			double cnt = phrsDocFreqs.getCount(phrs);
			if (cnt > 0) {
				seedPhrsCnts.setCount(phrs, cnt);
			}
		}

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsDocFreqs.keySet());
		Indexer<String> seedPhrsIdxer = Generics.newIndexer(seedPhrsCnts.keySet());

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser");

		Set<String> stopwords = FileUtils.readStringHashSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(dc.getVocab(), stopwords);
		Vocab vocab = dc.getVocab();

		DenseMatrix X1 = getFeatureMatrix(E, phrsIdxer, vocab, wf);
		DenseMatrix X2 = getFeatureMatrix(E, seedPhrsIdxer, vocab, wf);

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

		EmbeddingConceptBiasEstimatorOld pre = new EmbeddingConceptBiasEstimatorOld(phrsIdxer, X1, seedPhrsIdxer, X2,
				pWeights);

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

	private DenseMatrix Y;

	public EmbeddingConceptBiasEstimatorOld(Indexer<String> phrsIdxer, DenseMatrix X, Indexer<String> seedIdxer,
			DenseMatrix Y, DenseVector phrsTfidfs) {
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
			fs.add(tpe.submit(new Worker(row_cnt, timer)));
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
