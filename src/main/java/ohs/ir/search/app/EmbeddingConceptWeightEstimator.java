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
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
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
public class EmbeddingConceptWeightEstimator {

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

			while ((m = row_cnt.getAndIncrement()) < X1.rowSize()) {
				DenseVector x1 = X1.rowAt(m);

				if (x1.sum() == 0) {
					continue;
				}

				Counter<Integer> c = Generics.newCounter();

				for (int n = 0; n < X2.rowSize(); n++) {
					DenseVector x2 = X2.row(n);

					if (x2.sum() == 0) {
						continue;
					}

					double cosine = VectorMath.cosine(x1, x2);

					// if (cosine < min_cosine) {1
					// continue;
					// }

					c.incrementCount(n, cosine);
				}

				synchronized (phrsWeights) {
					phrsWeights.setCount(idxer1.getObject(m), c.average());
				}

				// System.out.printf("%s, %s\n", phrsIdxer.getObject(m),
				// VectorUtils.toCounter(c, phrsIdxer));

				int prog = BatchUtils.progress(m + 1, X1.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, m + 1, X1.rowSize(), timer.stop());
				}
			}

			return null;
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

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsCnts.keySet());
		Indexer<String> seedPhrsIdxer = Generics.newIndexer(seedPhrsCnts.keySet());

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser");

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(dc.getVocab(), stopwords);
		Vocab vocab = dc.getVocab();

		System.out.println("get phrase embeddings");

		DenseMatrix X1 = new DenseMatrix(phrsIdxer.size(), E.colSize());

		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);
			DenseVector x = X1.row(p);
			int cnt = 0;
			double idf_sum = 0;
			List<String> words = StrUtils.split(phrs);

			for (String word : words) {
				int w = vocab.indexOf(word);

				if (wf.filter(w)) {
					continue;
				}

				double idf = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w));
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

			int prog = BatchUtils.progress(p + 1, phrsIdxer.size());

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d]\n", prog, p + 1, phrsIdxer.size());
			}
		}

		DenseMatrix X2 = new DenseMatrix(seedPhrsIdxer.size(), E.colSize());

		for (int p = 0; p < seedPhrsIdxer.size(); p++) {
			String phrs = seedPhrsIdxer.getObject(p);
			DenseVector x = X2.row(p);
			int cnt = 0;
			double idf_sum = 0;
			List<String> words = StrUtils.split(phrs);

			for (String word : words) {
				int w = vocab.indexOf(word);

				if (wf.filter(w)) {
					continue;
				}

				double idf = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(w));
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

			int prog = BatchUtils.progress(p + 1, phrsIdxer.size());

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d]\n", prog, p + 1, phrsIdxer.size());
			}
		}

		EmbeddingConceptWeightEstimator pre = new EmbeddingConceptWeightEstimator(phrsIdxer, X1, seedPhrsIdxer, X2);

		pre.setThreadSize(7);
		// pre.setMinCosine(0.8);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phr_weight_emb.txt");

		System.out.println("process ends.");
	}

	private double min_cosine = 0.3;

	private Indexer<String> idxer1;

	private Indexer<String> idxer2;

	private int thread_size = 5;

	private DenseMatrix X1;

	private DenseMatrix X2;

	public EmbeddingConceptWeightEstimator(Indexer<String> idxer1, DenseMatrix X1, Indexer<String> idxer2,
			DenseMatrix X2) {
		this.idxer1 = idxer1;
		this.X1 = X1;
		this.idxer1 = idxer1;
		this.X2 = X2;
	}

	private Counter<String> phrsWeights;

	public void estimate(String outFileName) throws Exception {
		Timer timer = Timer.newTimer();

		phrsWeights = Generics.newCounter();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger row_cnt = new AtomicInteger(0);

		System.out.println("get similarity matrix");

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(row_cnt, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		tpe.shutdown();

		FileUtils.writeStringCounterAsText(outFileName, phrsWeights);

	}

	public void setMinCosine(double min_cosine) {
		this.min_cosine = min_cosine;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
