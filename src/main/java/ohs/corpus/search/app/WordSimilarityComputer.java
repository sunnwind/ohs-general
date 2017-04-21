package ohs.corpus.search.app;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.search.index.WordFilter;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.medical.general.MIRPath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class WordSimilarityComputer {

	public class Worker implements Callable<Map<Integer, SparseVector>> {
		private DenseMatrix M;

		public Worker(DenseMatrix M) {
			super();
			this.M = M;
		}

		@Override
		public Map<Integer, SparseVector> call() throws Exception {
			int l1 = 0;

			Map<Integer, SparseVector> ret = Generics.newHashMap();

			while ((l1 = row_cnt.getAndIncrement()) < ws.size()) {
				int w1 = ws.get(l1);
				DenseVector e1 = M.row(w1);
				Counter<Integer> c = Generics.newCounter();

				for (int l2 = l1 + 1; l2 < ws.size(); l2++) {
					int w2 = ws.get(l2);
					DenseVector e2 = M.row(w2);
					double cosine = VectorMath.dotProduct(e1, e2);

					if (cosine >= cutoff_cosine) {
						c.setCount(w2, cosine);
					}
				}

				if (c.size() > 0) {
					SparseVector sv = VectorUtils.toSparseVector(c);
					ret.put(w1, sv);

					// System.out.printf("%s\t%s\n", vocab.getObject(w1), VectorUtils.toCounter(sv, vocab));

				}

				if (l1 > 0 && l1 % 1000 == 0 || l1 == ws.size()) {
					System.out.printf("[%d/%d, %s]\n", l1, ws.size(), timer.stop());
				}
			}

			return ret;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2014_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		WordSimilarityComputer wsc = new WordSimilarityComputer(MIRPath.TREC_CDS_2014_DIR + "glove_model_raf.ser",
				MIRPath.TREC_CDS_2014_DIR + "cosine_word.ser.gz");
		wsc.setWordFilter(ds.getWordFilter());
		wsc.setVocab(ds.getVocab());
		wsc.setThreadSize(1);
		wsc.setCutoffCount(100);
		wsc.setCutoffCosine(0.7);

		wsc.compute();

		System.out.println("process ends.");
	}

	private Timer timer;

	private double cutoff_cosine = 0.7;

	private double cutoff_cnt = 100;

	private String inFileName;

	private String outFileName;

	private int thread_size = 10;

	private IntegerArray ws;

	private WordFilter filter;

	private AtomicInteger row_cnt;

	private Vocab vocab;

	public WordSimilarityComputer(String inFileName, String outFileName) {
		this.inFileName = inFileName;
		this.outFileName = outFileName;
	}

	public void compute() throws Exception {
		timer = Timer.newTimer();

		DenseMatrix M = new RandomAccessDenseMatrix(inFileName).rowsAsMatrix();

		ws = new IntegerArray(vocab.size());

		for (int w = 0; w < vocab.size(); w++) {
			if (!filter.filter(w) && vocab.getCount(w) >= cutoff_cnt) {
				ws.add(w);
			}
		}

		ws.trimToSize();

		// ws = ArrayUtils.range(E.rowSize());

		// ArrayUtils.shuffle(ws);

		row_cnt = new AtomicInteger(0);

		List<Future<Map<Integer, SparseVector>>> fs = Generics.newArrayList(thread_size);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		for (int i = 0; i < thread_size; i++) {
			Worker worker = new Worker(M);
			fs.add(tpe.submit(worker));
		}

		Map<Integer, SparseVector> res = Generics.newHashMap(M.rowSize());

		for (Future<Map<Integer, SparseVector>> f : fs) {
			res.putAll(f.get());
		}

		tpe.shutdown();

		new SparseMatrix(res).writeObject(outFileName);

	}

	public void setCutoffCosine(double cutoff_cosine) {
		this.cutoff_cosine = cutoff_cosine;
	}

	public void setCutoffCount(double cutoff_cnt) {
		this.cutoff_cnt = cutoff_cnt;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	public void setVocab(Vocab vocab) {
		this.vocab = vocab;
	}

	public void setWordFilter(WordFilter filter) {
		this.filter = filter;
	}

}
