package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.Set;

import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.math.ArrayMath;
import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer.Type;
import ohs.ml.neuralnet.layer.ConvolutionalLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.MaxPoolingLayer;
import ohs.ml.neuralnet.layer.MultiWindowConvolutionalLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Triple;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class Apps {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// testMNIST();
		// testCharRNN();
		testNER();

		// testSentenceClassification();

		System.out.println("process ends.");
	}

	public static void testCharRNN() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(100);
		param.setHiddenSize(50);
		param.setOutputSize(10);
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.01);
		param.setThreadSize(3);
		param.setBpttSize(10);

		// Triple<IntegerArrayMatrix, IntegerArrayMatrix, Vocab> train =
		// DataReader.readCapitalData(300);
		Triple<IntegerMatrix, IntegerMatrix, Vocab> data = DataReader.readLines("../../data/ml_data/warpeace_input.txt",
				1000);

		IntegerMatrix X = new IntegerMatrix();
		IntegerMatrix Y = new IntegerMatrix();
		IntegerMatrix Xt = new IntegerMatrix();
		IntegerMatrix Yt = new IntegerMatrix();
		Vocab vocab = data.getThird();

		{

			IntegerMatrix X_ = data.getFirst();
			IntegerMatrix Y_ = data.getSecond();

			double test_portion = 0.3;
			int test_size = (int) (1f * X_.size() * test_portion);

			for (int i = 0; i < X_.size(); i++) {
				if (i < test_size) {
					Xt.add(X_.get(i));
					Yt.add(Y_.get(i));
				} else {
					X.add(X_.get(i));
					Y.add(Y_.get(i));
				}
			}
		}

		Indexer<String> labelIdxer = Generics.newIndexer(vocab.getObjects());

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		int pad_label = vocab.getIndex("<PAD>");

		System.out.printf("data size: [%d -> %d]\n", size1, size2);
		System.out.printf("max len: [%d]\n", max_len);
		System.out.printf("min len: [%d]\n", min_len);

		int vocab_size = vocab.size();
		int input_size = vocab_size;
		int emb_size = 200;
		int l1_size = 100;
		int l2_size = 20;
		int output_size = vocab.size();
		int type = 4;

		System.out.println(vocab.info());

		NeuralNet nn = new NeuralNet(labelIdxer, vocab);

		if (type == 1) {
			nn.add(new EmbeddingLayer(vocab_size, emb_size, false));
			nn.add(new FullyConnectedLayer(emb_size, l1_size));
			nn.add(new NonlinearityLayer(new Tanh()));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, l2_size));
			nn.add(new NonlinearityLayer(new Tanh()));
			nn.add(new FullyConnectedLayer(l2_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 2) {
			// nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			// nn.add(new RnnLayer(embedding_size, l1_size, param.getBpttSize(),
			// new Tanh()));

			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			nn.add(new RnnLayer(emb_size, l1_size, param.getBpttSize(), new Tanh()));
			nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 3) {
			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			nn.add(new LstmLayer(emb_size, l1_size, new Tanh()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 4) {
			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			// nn.add(new GruLayer(embedding_size, l1_size, new Tanh()));
			// nn.add(new RnnLayer(embedding_size, l1_size, param.getBpttSize(), new
			// ReLU()));
			// nn.add(new LstmLayer(embedding_size, l1_size, new ReLU()));
			nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size, param.getBpttSize(), new ReLU()));

			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		}
	}

	public static void testMNIST() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(100);
		param.setHiddenSize(50);
		param.setOutputSize(10);
		param.setBatchSize(Integer.MAX_VALUE);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(8);
		param.setGradientClipCutoff(5);
		param.setOptimizerType(OptimizerType.RMSPROP);

		Pair<SparseMatrix, IntegerArray> train = DataReader.readFromSvmFormat("../../data/ml_data/mnist.txt");
		Pair<SparseMatrix, IntegerArray> test = DataReader.readFromSvmFormat("../../data/ml_data/mnist.t.txt");

		DenseMatrix X = train.getFirst().toDenseMatrix();
		IntegerArray Y = train.getSecond();

		DenseMatrix Xt = test.getFirst().toDenseMatrix(test.getFirst().rowSize(), X.colSize());
		IntegerArray Yt = test.getSecond();

		Set<String> labels = Generics.newTreeSet();
		for (int y : Y) {
			labels.add(y + "");
		}

		Indexer<String> labelIdxer = Generics.newIndexer(labels);

		int vocab_size = X.colSize();
		int l1_size = 200;
		int l2_size = 50;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet(labelIdxer, null);

		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}

	public static void testNER() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setBatchSize(5);
		param.setIsFullSequenceBatch(true);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(5);
		param.setBpttSize(10);
		param.setOptimizerType(OptimizerType.ADAM);
		param.setGradientClipCutoff(5);

		IntegerMatrix X = null;
		IntegerMatrix Y = null;
		Vocab vocab = null;
		Indexer<String> labelIdxer = null;
		IntegerMatrix Xt = null;
		IntegerMatrix Yt = null;

		{

			Object[] objs = DataReader.readNerTrainData("../../data/ml_data/conll2003.bio2/train.dat",
					"../../data/ml_data/conll2003.bio2/test.dat");
			X = (IntegerMatrix) objs[0];
			Y = (IntegerMatrix) objs[1];
			vocab = (Vocab) objs[2];
			labelIdxer = (Indexer<String>) objs[3];
		}

		{
			Object[] objs = DataReader.readNerTestData("../../data/ml_data/conll2003.bio2/test.dat", vocab, labelIdxer);
			Xt = (IntegerMatrix) objs[0];
			Yt = (IntegerMatrix) objs[1];
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		System.out.println(vocab.info());
		System.out.println(labelIdxer.info());
		System.out.println(X.sizeOfEntries());

		int voc_size = vocab.size();
		int emb_size = 50;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;

		NeuralNet nn = new NeuralNet(labelIdxer, vocab);

		if (type == 0) {
		} else if (type == 2) {
			nn.add(new EmbeddingLayer(voc_size, emb_size, true));
			nn.add(new DropoutLayer());
			nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size, param.getBpttSize(), new ReLU()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);

			IntegerArray locs = new IntegerArray(ArrayUtils.range(X.size()));

			int group_size = 1000;
			int[][] rs = BatchUtils.getBatchRanges(X.size(), group_size);

			int max_iters = 1000;

			for (int u = 0; u < max_iters; u++) {
				ArrayUtils.shuffle(locs.values());

				for (int i = 0; i < rs.length; i++) {
					System.out.printf("iters [%d/%d], batches [%d/%d]\n", u + 1, max_iters, i + 1, rs.length);
					for (int j = 0; j < rs.length; j++) {
						int[] r = rs[j];
						int range_size = r[1] - r[0];
						IntegerMatrix Xm = new IntegerMatrix(range_size);
						IntegerMatrix Ym = new IntegerMatrix(range_size);

						for (int k = r[0]; k < r[1]; k++) {
							int loc = locs.get(k);
							Xm.add(X.get(loc));
							Ym.add(Y.get(loc));
						}
						trainer.train(Xm, Ym, Xt, Yt, 1);
					}
				}
			}

			trainer.finish();
		}
	}

	public static void testSentenceClassification() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(100);
		param.setHiddenSize(50);
		param.setOutputSize(10);
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.01);
		param.setThreadSize(5);
		param.setBpttSize(10);

		IntegerMatrix X = new IntegerMatrix();
		IntegerArray Y = new IntegerArray();
		IntegerMatrix Xt = new IntegerMatrix();
		IntegerArray Yt = new IntegerArray();

		Vocab vocab = new Vocab();
		Indexer<String> labelIdxer = Generics.newIndexer();

		{
			List<String> sents = Generics.newLinkedList();

			for (String s : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.pos")) {
				sents.add(s + "\tpos");
			}

			for (String s : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.neg")) {
				sents.add(s + "\tneg");
			}

			IntegerMatrix M = new IntegerMatrix();
			IntegerArray N = new IntegerArray();

			for (int i = 0; i < sents.size(); i++) {
				String s = sents.get(i);
				String[] ps = s.split("\t");
				String sentiment = ps[1];
				int label = sentiment.equals("pos") ? 0 : 1;
				M.add(label, i);
				N.add(label);
			}

			IntegerMatrix T = DataSplitter.splitGroups(M, new int[] { 5000, 500 });

			for (int i = 0; i < T.size(); i++) {
				IntegerArray L = T.get(i);
				for (int loc : L) {
					String[] ps = sents.get(loc).split("\t");
					String sent = ps[0];
					String sentiment = ps[1];
					int label = N.get(loc);

					List<Integer> ws = vocab.getIndexes(StrUtils.split(sent));
					if (i == 0) {
						X.add(new IntegerArray(ws));
						Y.add(N.get(loc));
					} else {
						Xt.add(new IntegerArray(ws));
						Yt.add(N.get(loc));
					}
				}
			}
		}

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		int pad_label = vocab.getIndex("<PAD>");

		System.out.printf("data size: [%d -> %d]\n", size1, size2);
		System.out.printf("max len: [%d]\n", max_len);
		System.out.printf("min len: [%d]\n", min_len);

		int vocab_size = vocab.size();
		int input_size = vocab_size;
		int emb_size = 200;
		int l1_size = 100;
		int l2_size = 20;
		int output_size = 2;
		int type = 0;

		System.out.println(vocab.info());

		NeuralNet nn = new NeuralNet(labelIdxer, vocab);

		if (type == 0) {
			int num_filters = 100;
			int window_size = 2;
			int[] window_sizes = new int[] { 2 };

			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));

			// nn.add(new MultiWindowConvolutionalLayer(emb_size, window_sizes,
			// num_filters));
			nn.add(new ConvolutionalLayer(emb_size, window_size, num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		}

	}
}
