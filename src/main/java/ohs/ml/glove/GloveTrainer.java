package ohs.ml.glove;

import java.io.File;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * http://www.foldl.me/2014/glove-python/
 * 
 * @author ohs
 */
public class GloveTrainer {

	public class Worker implements Callable<Double> {

		private int[] cw_locs;

		private int[] range;

		private SparseMatrix C;

		private double eps = 0.00000001;

		private double beta1 = 0.9;

		private double beta2 = 0.999;

		private double learn_rate = param.getLearningRate();

		private double alpha = param.getAlpha();

		private double max_x = param.getMaxX();

		public Worker() {

		}

		@Override
		public Double call() throws Exception {
			double cost = 0;
			double cocnt = 0;
			double diff = 0;
			double fdiff = 0;
			double dx1 = 0;
			double dx2 = 0;
			double tmp1 = 0;
			double tmp2 = 0;

			int hidden_size = M.getW1().colSize();
			int center_w = 0;
			int ctx_w = 0;
			int ctw_loc = 0;
			int[] ctw_locs = null;

			if (use_adam) {
				for (int i = range[0]; i < range[1]; i++) {
					SparseVector row = C.rowAt(cw_locs[i]);
					center_w = C.indexAt(cw_locs[i]);

					DenseVector w1 = M.getW1().row(center_w);
					DenseVector b1 = M.getB1();

					DenseVector rw1 = R1.getW1().row(center_w);
					DenseVector rw11 = R2.getW1().row(center_w);

					ctw_locs = ArrayUtils.range(row.size());

					ArrayUtils.shuffle(ctw_locs);

					for (int j = 0; j < ctw_locs.length; j++) {
						ctw_loc = ctw_locs[j];
						ctx_w = row.indexAt(ctw_loc);
						cocnt = row.valueAt(ctw_loc);

						DenseVector w2 = M.getW2().row(ctx_w);
						DenseVector b2 = M.getB2();
						DenseVector rw2 = R1.getW2().row(ctx_w);
						DenseVector rw22 = R2.getW2().row(ctx_w);

						diff = VectorMath.dotProduct(w1, w2) + b1.value(center_w) + b2.value(ctx_w) - Math.log(cocnt);
						fdiff = cocnt < max_x ? Math.pow(cocnt / max_x, alpha) * diff : diff;
						cost += 0.5 * diff * fdiff;

						for (int k = 0; k < hidden_size; k++) {
							dx1 = fdiff * w2.value(k);
							tmp1 = ArrayMath.addAfterMultiply(rw1.value(k), beta1, dx1);
							tmp2 = ArrayMath.addAfterMultiply(rw11.value(k), beta2, Math.pow(dx1, 2));
							rw1.set(k, tmp1);
							rw11.set(k, tmp2);
							tmp1 /= (1 - beta1);
							tmp2 /= (1 - beta2);
							w1.set(k, w1.value(k) - learn_rate / Math.sqrt(tmp2 + eps) * tmp1);

							dx2 = fdiff * w1.value(k);
							tmp1 = ArrayMath.addAfterMultiply(rw2.value(k), beta1, dx2);
							tmp2 = ArrayMath.addAfterMultiply(rw22.value(k), beta2, Math.pow(dx2, 2));
							rw2.set(k, tmp1);
							rw22.set(k, tmp2);
							tmp1 /= (1 - beta1);
							tmp2 /= (1 - beta2);
							w2.set(k, w2.value(k) - learn_rate / Math.sqrt(tmp2 + eps) * tmp1);
						}

						DenseVector rb1 = R1.getB1();
						DenseVector rb11 = R2.getB1();

						dx1 = fdiff;
						tmp1 = ArrayMath.addAfterMultiply(rb1.value(center_w), beta1, dx1);
						tmp2 = ArrayMath.addAfterMultiply(rb11.value(center_w), beta2, Math.pow(dx1, 2));
						rb1.set(center_w, tmp1);
						rb11.set(center_w, tmp2);
						tmp1 /= (1 - beta1);
						tmp2 /= (1 - beta2);
						b1.set(center_w, b1.value(center_w) - learn_rate / Math.sqrt(tmp2 + eps) * tmp1);

						DenseVector rb2 = R1.getB2();
						DenseVector rb22 = R2.getB2();

						dx2 = fdiff;
						tmp1 = ArrayMath.addAfterMultiply(rb2.value(ctx_w), beta1, dx2);
						tmp2 = ArrayMath.addAfterMultiply(rb22.value(ctx_w), beta2, Math.pow(dx2, 2));
						rb2.set(ctx_w, tmp1);
						rb22.set(ctx_w, tmp2);
						tmp1 /= (1 - beta1);
						tmp2 /= (1 - beta2);
						b2.set(ctx_w, b2.value(ctx_w) - learn_rate / Math.sqrt(tmp2 + eps) * tmp1);
					}
				}
			} else {
				for (int i = range[0]; i < range[1]; i++) {
					SparseVector row = C.rowAt(cw_locs[i]);
					center_w = C.indexAt(cw_locs[i]);

					DenseVector w1 = M.getW1().row(center_w);
					DenseVector b1 = M.getB1();
					DenseVector rw1 = R1.getW1().row(center_w);

					ctw_locs = ArrayUtils.range(row.size());

					ArrayUtils.shuffle(ctw_locs);

					for (int j = 0; j < ctw_locs.length; j++) {
						ctw_loc = ctw_locs[j];
						ctx_w = row.indexAt(ctw_loc);
						cocnt = row.valueAt(ctw_loc);

						DenseVector w2 = M.getW2().row(ctx_w);
						DenseVector b2 = M.getB2();
						DenseVector rw2 = R1.getW2().row(ctx_w);

						diff = VectorMath.dotProduct(w1, w2) + b1.value(center_w) + b2.value(ctx_w) - Math.log(cocnt);
						fdiff = cocnt < max_x ? Math.pow(cocnt / max_x, alpha) * diff : diff;
						cost += 0.5 * diff * fdiff;

						for (int k = 0; k < hidden_size; k++) {
							dx1 = fdiff * w2.value(k);
							dx2 = fdiff * w1.value(k);

							rw1.add(k, Math.pow(dx1, 2));
							rw2.add(k, Math.pow(dx2, 2));

							w1.set(k, w1.value(k) - learn_rate / Math.sqrt(rw1.value(k) + eps) * dx1);
							w2.set(k, w2.value(k) - learn_rate / Math.sqrt(rw2.value(k) + eps) * dx2);
						}

						DenseVector rb1 = R1.getB1();
						DenseVector rb2 = R1.getB2();

						rb1.add(center_w, Math.pow(fdiff, 2));
						rb2.add(ctx_w, Math.pow(fdiff, 2));

						b1.set(center_w, b1.value(center_w) - learn_rate / Math.sqrt(rb1.value(center_w) + eps) * fdiff);
						b2.set(ctx_w, b2.value(ctx_w) - learn_rate / Math.sqrt(rb2.value(ctx_w) + eps) * fdiff);
					}
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

		String[] dataDirs = { KPPath.COL_DC_DIR, MIRPath.DATA_DIR + "merged/col/dc", MIRPath.OHSUMED_COL_DC_DIR,
				MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.WIKI_COL_DC_DIR, MIRPath.BIOASQ_COL_DC_DIR };
		int thread_size = 15;
		int hidden_size = 200;
		int max_iters = 100;
		int window_size = 10;
		int batch_size = 100;
		double learn_rate = 0.001;
		boolean use_adam = true;
		boolean read_all_files = false;
		boolean count = false;

		// if (use_adam) {
		// learn_rate = 0.001;
		// }

		for (int u = 0; u < dataDirs.length; u++) {
			if (u != 0) {
				continue;
			}

			File dataDir = new File(dataDirs[u]);
			File cocntDir = new File("/data1/ohs/tmp_cocnt/");
			File outFile = new File(dataDir.getParentFile().getParentFile(), "emb/glove.ser.gz");

			DocumentCollection dc = null;

			if (count) {
				CooccurrenceCounter cc = new CooccurrenceCounter(dataDir.getPath(), cocntDir.getPath(), null);
				cc.setWindowSize(window_size);
				cc.setCountThreadSize(5);
				cc.setSymmetric(true);
				cc.setOutputFileSize(4000);
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

			GloveTrainer trainer = new GloveTrainer();
			GloveModel M = trainer.train(param, vocab, cocntDir.getPath(), max_iters, read_all_files, use_adam);
			M.getAveragedModel().writeObject(outFile.getPath());

			FileUtils.deleteFilesUnder(cocntDir);
		}

		System.out.println("process ends.");
	}

	private GloveParam param;

	private GloveModel M;

	private GloveModel R1;

	private GloveModel R2;

	private boolean use_adam;

	public GloveTrainer() {

	}

	public GloveModel train(GloveParam param, Vocab vocab, String dataDir, int num_iters, boolean read_all_files, boolean use_adam)
			throws Exception {
		Timer timer1 = Timer.newTimer();

		this.param = param;
		this.use_adam = use_adam;

		List<File> files = FileUtils.getFilesUnder(dataDir);

		M = new GloveModel(param.getVocabSize(), param.getHiddenSize());
		M.init();

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

				double norm = VectorMath.normL2(M.getW());

				System.out.printf("%dth, cost: %f, time: %s (%s), norm: %f, learn-rate: %f\n", iters, cost, timer2.stop(), timer1.stop(),
						norm, param.getLearningRate());

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

					int[][] ranges = BatchUtils.getBatchRanges(C.rowSize(), C.rowSize() / Math.min(C.rowSize(), thread_size));

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

				double norm = VectorMath.normL2(M.getW());

				System.out.printf("%dth, cost: %f, time: %s (%s), norm: %f, learn-rate: %f\n", iters, cost, timer2.stop(), timer1.stop(),
						norm, param.getLearningRate());

				if (cost > old_cost) {
					break;
				}
				old_cost = cost;
			}
		}

		tpe.shutdown();

		return M;
	}

}
