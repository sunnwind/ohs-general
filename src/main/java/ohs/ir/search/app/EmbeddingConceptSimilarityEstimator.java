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
public class EmbeddingConceptSimilarityEstimator {

	class Worker implements Callable<CounterMap<Integer, Integer>> {

		private double min_cosine = 0.3;

		private AtomicInteger row_cnt;

		private Timer timer;

		private DenseMatrix X;

		public Worker(DenseMatrix X, double min_cosine, AtomicInteger range_cnt, Timer timer) {
			this.X = X;
			this.min_cosine = min_cosine;
			this.row_cnt = range_cnt;
			this.timer = timer;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {
			int m = 0;

			while ((m = row_cnt.getAndIncrement()) < X.rowSize()) {
				DenseVector x1 = X.rowAt(m);

				if (x1.sum() == 0) {
					continue;
				}

				Counter<Integer> c = Generics.newCounter();

				for (int n = m + 1; n < X.rowSize(); n++) {
					DenseVector x2 = X.row(n);

					if (x2.sum() == 0) {
						continue;
					}

					double cosine = VectorMath.cosine(x1, x2);

					if (cosine < min_cosine) {
						continue;
					}

					c.incrementCount(n, cosine);
				}

				if (c.size() > 0) {
					String phrs1 = phrsIdxer.getObject(m);
					List<String> res = Generics.newArrayList(c.size());

					for (int p2 : c.getSortedKeys()) {
						String phrs2 = phrsIdxer.getObject(p2);
						res.add(String.format("%s\t%s\t%f", phrs1, phrs2, c.getCount(p2)));
					}

					synchronized (writer) {
						writer.write(StrUtils.join("\n", res) + "\n");
					}
				}

				int prog = BatchUtils.progress(m, X.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s]\n", prog, m, X.rowSize(), timer.stop());
				}
			}

			return null;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsCnts = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt")) {
			String[] ps = line.split("\t");
			String phrs = ps[0];
			double tfidf = Double.parseDouble(ps[1]);
			double cnt = Double.parseDouble(ps[2]);
			double doc_freq = Double.parseDouble(ps[3]);
			phrsCnts.setCount(phrs, cnt);
		}

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsCnts.keySet());

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(MIRPath.TREC_CDS_2016_DIR + "emb/glove_ra.ser");

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter wf = new WordFilter(dc.getVocab(), stopwords);
		Vocab vocab = dc.getVocab();

		DenseMatrix X = new DenseMatrix(phrsIdxer.size(), E.colSize());

		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);
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
				x.summation();
				cnt++;
			}

			if (cnt > 0) {
				x.multiply(1d / cnt);
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

		EmbeddingConceptSimilarityEstimator pre = new EmbeddingConceptSimilarityEstimator(phrsIdxer, X);

		pre.setThreadSize(7);
		pre.setMinCosine(0.8);
		pre.estimate(MIRPath.DATA_DIR + "phrs/phrs_sim.txt");

		System.out.println("process ends.");
	}

	private double min_cosine = 0.3;

	private Indexer<String> phrsIdxer;

	private int thread_size = 5;

	private TextFileWriter writer;

	private TextFileWriter writer2;

	private DenseMatrix X;

	public EmbeddingConceptSimilarityEstimator(Indexer<String> phrsIdxer, DenseMatrix X) {
		this.phrsIdxer = phrsIdxer;
		this.X = X;
	}

	public void estimate(String outFileName) throws Exception {
		Timer timer = Timer.newTimer();

		writer = new TextFileWriter(outFileName);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger row_cnt = new AtomicInteger(0);

		System.out.println("get similarity matrix");

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(X, min_cosine, row_cnt, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}
		tpe.shutdown();

		writer.close();
	}

	public void setMinCosine(double min_cosine) {
		this.min_cosine = min_cosine;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
