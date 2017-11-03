package ohs.ml.glove;

import java.io.File;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.ml.neuralnet.com.ParameterUpdater;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * http://www.foldl.me/2014/glove-python/
 * 
 * @author ohs
 */
public class GloveTrainerNew {

	public class Worker implements Callable<Double> {

		private double alpha = param.getAlpha();

		private SparseMatrix C;

		private int[] cw_locs;

		private double learn_rate = param.getLearningRate();

		private double max_x = param.getMaxX();

		private int[] range;

		public Worker() {

		}

		@Override
		public Double call() throws Exception {
			double cost = 0;

			DenseMatrix Wm = new DenseMatrix();
			DenseMatrix dWm = new DenseMatrix();
			DenseMatrix R1m = new DenseMatrix();
			DenseMatrix R2m = new DenseMatrix();

			for (int i = range[0]; i < range[1]; i++) {
				SparseVector row = C.rowAt(cw_locs[i]);
				int center_w = C.indexAt(cw_locs[i]);

				DenseVector w1 = W.getW1().row(center_w);
				DenseVector b1 = W.getB1();

				DenseVector dw1 = dW.getW1().row(center_w);
				DenseVector db1 = dW.getB1();

				DenseVector rw11 = R1.getW1().row(center_w);
				DenseVector rw21 = R2.getW1().row(center_w);

				DenseVector rb11 = R1.getB1();
				DenseVector rb21 = R2.getB1();

				int[] ctw_locs = ArrayUtils.range(row.size());

				ArrayUtils.shuffle(ctw_locs);

				for (int j = 0; j < ctw_locs.length; j++) {
					int ctw_loc = ctw_locs[j];
					int ctx_w = row.indexAt(ctw_loc);
					double cocnt = row.valueAt(ctw_loc);

					DenseVector w2 = W.getW2().row(ctx_w);
					DenseVector b2 = W.getB2();

					DenseVector dw2 = dW.getW2().row(ctx_w);
					DenseVector db2 = dW.getB2();

					DenseVector rw12 = R1.getW2().row(ctw_loc);
					DenseVector rw22 = R2.getW2().row(ctw_loc);

					DenseVector rb12 = R1.getB2();
					DenseVector rb22 = R2.getB2();

					double diff = VectorMath.dotProduct(w1, w2) + b1.value(center_w) + b2.value(ctx_w)
							- Math.log(cocnt);
					double fdiff = cocnt < max_x ? Math.pow(cocnt / max_x, alpha) * diff : diff;
					cost += 0.5 * diff * fdiff;

					VectorMath.addAfterMultiply(w2, fdiff, dw1);
					VectorMath.addAfterMultiply(w1, fdiff, dw2);

					db1.setAll(0);
					db2.setAll(0);

					db1.add(center_w, fdiff);
					db2.add(ctx_w, fdiff);

					Wm.clear();
					dWm.clear();
					R1m.clear();
					R2m.clear();

					Wm.add(w1);
					Wm.add(w2);
					Wm.add(b1);
					Wm.add(b2);

					dWm.add(dw1);
					dWm.add(dw2);
					dWm.add(db1);
					dWm.add(db2);

					R1m.add(rw11);
					R1m.add(rw12);
					R1m.add(rb11);
					R1m.add(rb12);

					R2m.add(rw21);
					R2m.add(rw22);
					R2m.add(rb21);
					R2m.add(rb22);

					ParameterUpdater pu = new ParameterUpdater(Wm.toDenseTensor(), dWm.toDenseTensor(),
							R1m.toDenseTensor(), R2m.toDenseTensor());
					pu.setGradientClipCutoff(Double.MAX_VALUE);
					pu.setLearningRate(learn_rate);
					pu.update();
				}
			}

			return cost / (range[1] - range[0]);
		}

		public void set(SparseMatrix C, int[] cw_locs, int[] range) {
			this.C = C;
			this.cw_locs = cw_locs;
			this.range = range;
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// String[] dirs = { MIRPath.OHSUMED_COL_DIR,
		// MIRPath.CLEF_EH_2014_COL_DIR, MIRPath.TREC_GENO_2007_COL_DIR,
		// MIRPath.TREC_CDS_2014_COL_DIR, MIRPath.TREC_CDS_2016_COL_DIR,
		// MIRPath.WIKI_COL_DIR, MIRPath.BIOASQ_COL_DIR };

		// String[] dataDirs = { KPPath.COL_DC_DIR, MIRPath.DATA_DIR + "merged/col/dc",
		// MIRPath.OHSUMED_COL_DC_DIR,
		// MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.WIKI_COL_DC_DIR,
		// MIRPath.BIOASQ_COL_DC_DIR };

		// String[] dataDirs = { FNPath.NAVER_NEWS_COL_DC_DIR };
		String[] dataDirs = { MIRPath.OHSUMED_COL_DC_DIR };
		int thread_size = 15;
		int hidden_size = 200;
		int max_iters = 100;
		int window_size = 10;
		int batch_size = 100;
		double learn_rate = 0.001;
		boolean use_adam = true;
		boolean read_all_files = true;
		boolean count = false;

		// if (use_adam) {
		// learn_rate = 0.001;
		// }

		for (int u = 0; u < dataDirs.length; u++) {
			File dataDir = new File(dataDirs[u]);
			File cocntDir = new File("G:", "/cocnt/");
			File outFile = new File("G:", "emb/glove.ser.gz");

			DocumentCollection dc = null;

			if (count) {
				CooccurrenceCounter cc = new CooccurrenceCounter(dataDir.getPath(), cocntDir.getPath(), null);
				cc.setWindowSize(window_size);
				cc.setCountThreadSize(5);
				cc.setSymmetric(true);
				cc.setOutputFileSize(100);
				cc.setBatchSize(10000);
				cc.setMinWordCount(10);
				cc.count();
				dc = cc.getDocumentCollection();
			} else {
				dc = new DocumentCollection(dataDir.getPath());
			}

			Vocab vocab = dc.getVocab();

			GloveParam param = new GloveParam(vocab.size(), hidden_size);
			param.setThreadSize(thread_size);
			param.setLearnRate(learn_rate);

			GloveTrainerNew trainer = new GloveTrainerNew();
			GloveModel M = trainer.train(param, vocab, cocntDir.getPath(), max_iters, read_all_files, use_adam);
			M.getAveragedModel().writeObject(outFile.getPath());

			FileUtils.deleteFilesUnder(cocntDir);
		}

		System.out.println("process ends.");
	}

	private GloveModel dW;

	private GloveModel W;

	private GloveParam param;

	private GloveModel R1;

	private GloveModel R2;

	private boolean use_adam;

	public GloveTrainerNew() {

	}

	public GloveModel train(GloveParam param, Vocab vocab, String dataDir, int num_iters, boolean read_all_files,
			boolean use_adam) throws Exception {
		Timer timer1 = Timer.newTimer();

		this.param = param;
		this.use_adam = use_adam;

		List<File> files = FileUtils.getFilesUnder(dataDir);

		W = new GloveModel(param.getVocabSize(), param.getHiddenSize());
		W.init();

		dW = new GloveModel(param.getVocabSize(), param.getHiddenSize());
		R1 = new GloveModel(param.getVocabSize(), param.getHiddenSize());
		R2 = new GloveModel(param.getVocabSize(), param.getHiddenSize());

		int thread_size = param.getThreadSize();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Worker> workers = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			workers.add(new Worker());
		}

		if (read_all_files) {
			List<Integer> idxs = Generics.newLinkedList();
			List<SparseVector> rows = Generics.newLinkedList();

			for (int i = 0; i < files.size(); i++) {
				File file = files.get(i);
				SparseMatrix tmp = new SparseMatrix();
				tmp.readObject(file.getPath());

				for (int j = 0; j < tmp.rowSize(); j++) {
					int idx = tmp.indexAt(j);
					SparseVector row = tmp.rowAt(j);
					idxs.add(idx);
					rows.add(row);
				}
			}

			SparseMatrix C = new SparseMatrix(idxs, rows);

			int[] cw_locs = ArrayUtils.range(C.rowSize());

			int[][] ranges = BatchUtils.getBatchRanges(C.rowSize(), C.rowSize() / Math.min(C.rowSize(), thread_size));

			double old_cost = Double.MAX_VALUE;

			for (int iters = 1; iters <= num_iters; iters++) {
				Timer timer2 = Timer.newTimer();
				double cost = 0;

				ArrayUtils.shuffle(cw_locs);

				List<Future<Double>> fs = Generics.newArrayList();

				for (int j = 0; j < thread_size && j < ranges.length; j++) {
					Worker worker = workers.get(j);
					worker.set(C, cw_locs, ranges[j]);
					fs.add(tpe.submit(worker));
				}

				for (int j = 0; j < fs.size(); j++) {
					cost += fs.get(j).get();
				}
				fs.clear();

				double norm = VectorMath.normL2(W.getW());

				System.out.printf("%dth, cost: %f, time: %s (%s), norm: %f, learn-rate: %f\n", iters, cost,
						timer2.stop(), timer1.stop(), norm, param.getLearningRate());

				if (cost > old_cost) {
					break;
				}
				old_cost = cost;
			}
		} else {
			int[] file_locs = ArrayUtils.range(files.size());

			double old_cost = Double.MAX_VALUE;

			for (int iters = 1; iters <= num_iters; iters++) {
				Timer timer2 = Timer.newTimer();
				double cost = 0;

				ArrayUtils.shuffle(file_locs);

				for (int i = 0; i < files.size(); i++) {
					File file = files.get(file_locs[i]);
					SparseMatrix C = new SparseMatrix();
					C.readObject(file.getPath());

					int[] cw_locs = ArrayUtils.range(C.rowSize());

					ArrayUtils.shuffle(cw_locs);

					int[][] ranges = BatchUtils.getBatchRanges(C.rowSize(),
							C.rowSize() / Math.min(C.rowSize(), thread_size));

					List<Future<Double>> fs = Generics.newArrayList();

					for (int j = 0; j < thread_size && j < ranges.length; j++) {
						Worker worker = workers.get(j);
						worker.set(C, cw_locs, ranges[j]);
						fs.add(tpe.submit(worker));
					}

					for (int j = 0; j < fs.size(); j++) {
						cost += fs.get(j).get();
					}
					fs.clear();
				}

				double norm = VectorMath.normL2(W.getW());

				System.out.printf("%dth, cost: %f, time: %s (%s), norm: %f, learn-rate: %f\n", iters, cost,
						timer2.stop(), timer1.stop(), norm, param.getLearningRate());

				if (cost > old_cost) {
					break;
				}
				old_cost = cost;
			}
		}

		tpe.shutdown();

		return W;
	}

}
