package ohs.ml.word2vec;

import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.SentenceCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterUpdater;
import ohs.ml.word2vec.Word2VecParam.CostType;
import ohs.ml.word2vec.Word2VecParam.NetworkType;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * 
 * http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_I_The_Skip-Gram_Model.pdf
 * 
 * @author ohs
 */
public class Word2VecTrainer {

	public class Worker implements Callable<Integer> {
		private Word2VecModel G;

		private DenseMatrix tmp_dW;

		private DenseVector tmp_h = new DenseVector(param.getHiddenSize());

		private DenseVector tmp_v = new DenseVector(param.getVocabSize());

		private ParameterUpdater updater;

		public Worker() {
			G = new Word2VecModel(param.getVocabSize(), param.getHiddenSize());

			updater = new ParameterUpdater(M.getW(), G.getW(), R1.getW(), R2.getW(), param.getMiniBatchSize());
			updater.setUseBatchSizeScale(true);
		}

		@Override
		public Integer call() throws Exception {
			double eps = 0.00000001;
			int batch_size = param.getMiniBatchSize();
			int context_size = param.getContextSize();
			int cur_iter = 0;

			int correct_cnt = 0;
			int train_word_cnt = 0;
			int random_context_size = 0;
			int sloc = 0;
			int wloc = 0;
			int center_w = 0;
			double cost = 0;

			while ((cur_iter = iter_cnt.incrementAndGet()) <= max_iters) {
				Timer timer = Timer.newTimer();

				cost = 0;
				correct_cnt = 0;
				train_word_cnt = 0;

				for (int i = 0; i < batch_size; i++) {
					random_context_size = ArrayMath.random(1, context_size + 1);
					sloc = sc.getRandomSentLoc();
					wloc = sc.getRandomWordLoc(sloc);
					center_w = sc.getWord(sloc, wloc);

					String center_word = sc.getVocab().getObject(center_w);

					IntegerArray context_ws = sc.getContextWords(sloc, wloc, random_context_size);

					if (context_ws.size() == 0) {
						continue;
					}

					String[] words = sc.getVocab().getObjects(context_ws.clone().values());

					Pair<Double, Integer> res = null;

					if (param.getType() == NetworkType.CBOW) {
						res = CBOW(center_w, context_ws);
						train_word_cnt++;
					} else if (param.getType() == NetworkType.SKIP_GRAM) {
						res = Skipgram(center_w, context_ws);
						train_word_cnt += context_ws.size();
					}

					cost += (res.getFirst() / batch_size);
					correct_cnt += res.getSecond();
				}

				cost *= -1;

				updater.update();

				double acc = 1f * correct_cnt / train_word_cnt;

				double norm = VectorMath.normL2(M.getW());

				if (cur_iter % 200 == 0) {
					System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s, norm: %f, learn-rate: %f\n", cur_iter, cost, acc,
							correct_cnt, train_word_cnt, timer.stop(), norm, learning_rate);
				}
			}

			return 0;
		}

		public Pair<Double, Integer> CBOW(int center_w, IntegerArray context_ws) {
			DenseVector predicted = tmp_h;
			predicted.setAll(0);

			for (int c : context_ws.clone().values()) {
				VectorMath.add(predicted, M.getWxh().row(c), predicted);
			}

			Pair<Double, Integer> res = null;
			if (param.getCostType() == CostType.NEG_SAMPLE_COST) {
				res = negativeSamplingCost(center_w, context_ws, predicted, center_w, param.getNegSampleSize());
			} else if (param.getCostType() == CostType.SOFTMAX) {
				res = softmaxCost(center_w, context_ws, predicted, center_w);
			}

			return res;
		}

		public Pair<Double, Integer> negativeSamplingCost(int center_w, IntegerArray context_ws, DenseVector predicted, int target_w,
				int sample_size) {

			int data_size = sample_size + 1;
			IntegerArray ws = new IntegerArray(data_size);
			DenseVector labels = new DenseVector(data_size);
			List<DenseVector> rows1 = Generics.newArrayList(data_size);
			List<DenseVector> rows2 = Generics.newArrayList(data_size);

			ws.add(target_w);
			labels.add(0, 1);
			rows1.add(M.getWyh().get(target_w));
			rows2.add(G.getWyh().get(target_w));

			for (int i = 0; i < sample_size; i++) {
				int w = sc.sampleWord();
				while (w == target_w) {
					w = sc.sampleWord();
				}
				ws.add(w);
				rows1.add(M.getWyh().get(w));
				rows2.add(G.getWyh().get(w));
				labels.add(i + 1, -1);
			}

			DenseMatrix Wyh = new DenseMatrix(rows1);
			DenseMatrix dWyh = new DenseMatrix(rows2);

			DenseVector yh = new DenseVector(data_size);

			VectorMath.product(Wyh, predicted, yh, false);

			VectorMath.multiply(yh, labels, yh);

			VectorMath.sigmoid(yh, yh);

			double cost = VectorMath.sumAfterLog(yh);

			int pred = yh.argMax();

			int correct_cnt = pred == 0 ? 1 : 0;

			DenseVector delta = yh;

			VectorMath.add(delta, -1, delta);

			VectorMath.multiply(labels, delta, delta);

			if (tmp_dW == null || tmp_dW.rowSize() < ws.size()) {
				tmp_dW = new DenseMatrix(data_size, param.getHiddenSize());
			}

			DenseMatrix dWyh0 = tmp_dW.rowsAsMatrix(data_size);

			VectorMath.outerProduct(delta, predicted, dWyh0, false);

			VectorMath.add(dWyh0, dWyh);

			DenseVector dwxh0 = tmp_h;

			VectorMath.product(delta, Wyh, dwxh0, false);

			if (param.getType() == NetworkType.SKIP_GRAM) {
				DenseVector dwxh = G.getWxh().row(center_w);
				VectorMath.add(dwxh0, dwxh);
			} else if (param.getType() == NetworkType.CBOW) {
				for (int w : context_ws.clone().values()) {
					DenseVector dwxh = G.getWxh().row(w);
					VectorMath.add(dwxh0, dwxh);
				}
			}

			return Generics.newPair(cost, correct_cnt);
		}

		public Pair<Double, Integer> Skipgram(int center_w, IntegerArray context_ws) {
			double cost = 0;
			int correct_cnt = 0;
			DenseVector predicted = M.getWxh().row(center_w);

			for (int i = 0; i < context_ws.size(); i++) {
				int target_w = context_ws.get(i);
				Pair<Double, Integer> res = null;
				if (param.getCostType() == CostType.SOFTMAX) {
					res = softmaxCost(center_w, context_ws, predicted, target_w);
				} else if (param.getCostType() == CostType.NEG_SAMPLE_COST) {
					res = negativeSamplingCost(center_w, context_ws, predicted, target_w, param.getNegSampleSize());
				}
				cost += res.getFirst();
				correct_cnt += res.getSecond();
			}
			return Generics.newPair(cost, correct_cnt);
		}

		public Pair<Double, Integer> softmaxCost(int center_w, IntegerArray context_ws, DenseVector x, int target_w) {
			DenseVector yh = tmp_v;
			yh.setAll(0);

			VectorMath.productRows(x.toDenseMatrix(), M.getWyh(), yh.toDenseMatrix(), false);
			VectorMath.softmax(yh, yh);

			double cost = Math.log(yh.value(target_w));

			int num_correct = yh.argMax() == target_w ? 1 : 0;

			DenseVector delta = yh;
			delta.add(target_w, -1);

			VectorMath.outerProduct(delta, x, G.getWyh(), true);

			DenseVector g_in = new DenseVector(param.getHiddenSize());
			VectorMath.product(delta, M.getWyh(), g_in, false);

			if (param.getType() == NetworkType.SKIP_GRAM) {
				DenseVector dw1 = G.getWxh().row(center_w);
				VectorMath.add(dw1, g_in, dw1);
			} else {
				for (int w : context_ws.clone().values()) {
					DenseVector dw1 = G.getWxh().row(w);
					VectorMath.add(dw1, g_in, dw1);
				}
			}

			return Generics.newPair(cost, num_correct);
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// List<String> docs = Generics.newArrayList();

		// {
		// TextFileReader reader = new TextFileReader(MIRPath.TREC_CDS_DIR +
		// "sents.txt.gz");
		// while (reader.hasNext()) {
		//
		// if (docs.size() == 200) {
		// break;
		// }
		//
		// String line = reader.next();
		// String[] parts = line.split("\t");
		//
		// parts = StrUtils.unwrap(parts);
		//
		// String content = StrUtils.join("\n", parts, 1, 4);
		//
		// docs.add(content.replace(StrUtils.LINE_REP, "\n"));
		// }
		// reader.close();
		// }

		// String modelFileName = MIRPath.TREC_CDS_DIR +
		// "word2vec_model.ser.gz";

		// if (!FileUtils.exists(modelFileName)) {
		// SentenceCollectionCreator scc = new SentenceCollectionCreator();
		// scc.setPreprocess(true);

		String dirPath = MIRPath.OHSUMED_DIR;
		String scFileName = dirPath + "sc.ser.gz";
		String modelFileName = dirPath + "word2vec_model.ser.gz";
		String wvFileName = dirPath + "word2vec_word_vecs.ser.gz";

		DocumentCollection ldc = new DocumentCollection(MIRPath.OHSUMED_DIR + "/col/dc/");
		SentenceCollection sc = new SentenceCollection();

		int size1 = sc.sizeOfEntries();

		for (IntegerArray sent : sc) {
			IntegerArray tmp = new IntegerArray();

			for (int i = 0; i < sent.size(); i++) {
				int w = sent.get(i);
				if (w >= 2) {
					tmp.add(w);
				}
			}

			sent.clear();

			for (int i = 0; i < tmp.size(); i++) {
				sent.add(tmp.get(i));
			}
		}

		int size2 = sc.sizeOfEntries();

		System.out.printf("size: %d->%d\n", size1, size2);

		sc.makeSampleTable(sc.getVocab().size() * 3);

		Word2VecParam param = new Word2VecParam(NetworkType.SKIP_GRAM, CostType.NEG_SAMPLE_COST, sc.getVocab().size(), 200, 10);
		param.setContextSize(5);
		param.setLearningRate(0.001);
		param.setNegSampleSize(10);
		param.setMiniBatchSize(50);
		param.setNumThreads(5);
		param.setHiddenSize(200);

		Word2VecTrainer trainer = new Word2VecTrainer();
		Word2VecModel M = trainer.train(param, sc, 10000);
		M.writeObject(modelFileName);

		System.out.println(sc.getVocab().getObjects());

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);

		WordSearcher.interact(new WordSearcher(ldc.getVocab(), M.getAveragedModel(), stopwords));

		System.out.println("process ends.");
	}

	private AtomicInteger iter_cnt = new AtomicInteger(0);

	private double learning_rate;

	private Word2VecModel M;

	private int max_iters;

	private Word2VecParam param;

	private Word2VecModel R1;

	private Word2VecModel R2;

	private SentenceCollection sc;

	public Word2VecTrainer() {

	}

	public Word2VecModel train(Word2VecParam param, SentenceCollection sc, int max_iters) throws Exception {
		this.param = param;
		this.sc = sc;
		this.learning_rate = param.getLearningRate();
		this.max_iters = max_iters;

		M = new Word2VecModel(param.getVocabSize(), param.getHiddenSize());
		R1 = new Word2VecModel(param.getVocabSize(), param.getHiddenSize());
		R2 = new Word2VecModel(param.getVocabSize(), param.getHiddenSize());

		M.init();

		System.out.println(M.info());

		int thread_size = param.getNumThreads();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker()));
		}

		for (int j = 0; j < fs.size(); j++) {
			fs.get(j).get();
		}

		tpe.isShutdown();

		return M;
	}

}
