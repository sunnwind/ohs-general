package ohs.ml.neuralnet.com;

import java.util.Set;

import ohs.matrix.DenseMatrix;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.BiRnnLayer;
import ohs.ml.neuralnet.layer.ConvolutionalLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.layer.WindowLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Triple;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class Apps {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// testMIST();
		testCharRNN();
		// testNER();

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
		param.setBpttSize(0);

		// Triple<IntegerArrayMatrix, IntegerArrayMatrix, Vocab> train =
		// DataReader.readCapitalData(300);
		Triple<IntegerArrayMatrix, IntegerArrayMatrix, Vocab> data = DataReader.readLines("../../data/ml_data/warpeace_input.txt", 1000);

		IntegerArrayMatrix X = new IntegerArrayMatrix();
		IntegerArrayMatrix Y = new IntegerArrayMatrix();
		IntegerArrayMatrix Xt = new IntegerArrayMatrix();
		IntegerArrayMatrix Yt = new IntegerArrayMatrix();
		Vocab vocab = data.getThird();

		{

			IntegerArrayMatrix rX = data.getFirst();
			IntegerArrayMatrix rY = data.getSecond();

			double test_portion = 0.3;
			int test_size = (int) (1f * rX.size() * test_portion);

			for (int i = 0; i < rX.size(); i++) {
				// if (i < test_size) {
				// tX.add(rX.get(i));
				// tY.add(rY.get(i));
				// } else {
				X.add(rX.get(i));
				Y.add(rY.get(i));
				// }
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
		int embedding_size = 200;
		int l1_size = 100;
		int l2_size = 20;
		int output_size = vocab.size();
		int type = 3;

		System.out.println(vocab.info());

		NeuralNet nn = new NeuralNet();

		if (type == 0) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			nn.add(new ConvolutionalLayer(embedding_size, new int[] { 3 }, 100));
			// nn.add(new WindowLayer(window_size, embedding_size));
			// nn.add(new FullyConnectedLayer((2 * window_size + 1) *
			// embedding_size, l1_size));
			// nn.add(new BatchNormalizationLayer(input_size));

			nn.add(new NonlinearityLayer(nn.last().getOutputSize(), new Tanh()));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(nn.last().getOutputSize(), l2_size));
			nn.add(new NonlinearityLayer(l2_size, new Tanh()));
			nn.add(new FullyConnectedLayer(l2_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 1) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, false));
			nn.add(new FullyConnectedLayer(embedding_size, l1_size));
			nn.add(new NonlinearityLayer(l1_size, new Tanh()));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, l2_size));
			nn.add(new NonlinearityLayer(l2_size, new Tanh()));
			nn.add(new FullyConnectedLayer(l2_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 2) {
			// nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			// nn.add(new RnnLayer(embedding_size, l1_size, param.getBpttSize(),
			// new Tanh()));

			nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			nn.add(new RnnLayer(embedding_size, l1_size, param.getBpttSize(), new Tanh()));
			nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 3) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			nn.add(new LstmLayer(embedding_size, l1_size, new Tanh()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		} else if (type == 4) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			// nn.add(new GruLayer(embedding_size, l1_size, new Tanh()));
			nn.add(new BiRnnLayer(embedding_size, l1_size, param.getBpttSize(), new Tanh()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
			trainer.train(X, Y, null, null, 10000);
			trainer.finish();
		}
	}

	public static void testMIST() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(100);
		param.setHiddenSize(50);
		param.setOutputSize(10);
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(10);

		Pair<SparseMatrix, IntegerArray> train = DataReader.readFromSvmFormat("../../data/ml_data/mnist.txt");
		Pair<SparseMatrix, IntegerArray> test = DataReader.readFromSvmFormat("../../data/ml_data/mnist.t.txt");

		DenseMatrix X = train.getFirst().toDenseMatrix();
		IntegerArray Y = train.getSecond();

		DenseMatrix Xt = test.getFirst().toDenseMatrix(test.getFirst().rowSize(), X.colSize());
		IntegerArray Yt = test.getSecond();

		Set<Integer> labels = Generics.newHashSet();
		for (int y : Y) {
			labels.add(y);
		}

		int vocab_size = X.colSize();
		int l1_size = 100;
		int l2_size = 50;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(l1_size, new Tanh()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(l2_size, new Tanh()));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}

	public static void testNER() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(100);
		param.setHiddenSize(50);
		param.setOutputSize(10);
		param.setBatchSize(1);
		param.setLearnRate(0.001);
		param.setRegLambda(0.01);
		param.setThreadSize(1);
		param.setBpttSize(0);

		IntegerArrayMatrix X = null;
		IntegerArrayMatrix Y = null;
		Vocab vocab = null;
		Indexer<String> labelIndexer = null;
		IntegerArrayMatrix Xt = null;
		IntegerArrayMatrix Yt = null;

		{

			Object[] objs = DataReader.readNerTrainData("../../data/ml_data/conll2003.bio2/train.dat");
			X = (IntegerArrayMatrix) objs[0];
			Y = (IntegerArrayMatrix) objs[1];
			vocab = (Vocab) objs[2];
			labelIndexer = (Indexer<String>) objs[3];
		}

		{
			Object[] objs = DataReader.readNerTestData("../../data/ml_data/conll2003.bio2/test.dat", vocab, labelIndexer);
			Xt = (IntegerArrayMatrix) objs[0];
			Yt = (IntegerArrayMatrix) objs[1];
		}

		System.out.println(vocab.info());
		System.out.println(labelIndexer.info());
		System.out.println(X.sizeOfEntries());

		int vocab_size = vocab.size();
		int embedding_size = 200;
		int l1_size = 100;
		int l2_size = 20;
		int output_size = labelIndexer.size();
		int window_size = 5;
		int type = 2;

		NeuralNet nn = new NeuralNet();

		if (type == 0) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, false));
			// nn.add(new ConvolutionalLayer(embedding_size, new int[] { 2, 3 },
			// 100));
			nn.add(new WindowLayer(window_size, embedding_size));
			nn.add(new FullyConnectedLayer(nn.last().getOutputSize(), l1_size));
			nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new NonlinearityLayer(nn.last().getOutputSize(), new Tanh()));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(nn.last().getOutputSize(), output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), labelIndexer);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		} else if (type == 1) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			nn.add(new FullyConnectedLayer(embedding_size, l1_size));
			nn.add(new NonlinearityLayer(l1_size, new Tanh()));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, l2_size));
			nn.add(new NonlinearityLayer(l2_size, new Tanh()));
			nn.add(new FullyConnectedLayer(l2_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), labelIndexer);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		} else if (type == 2) {
			nn.add(new EmbeddingLayer(vocab_size, embedding_size, true));
			// nn.add(new BatchNormalizationLayer(embedding_size));
			nn.add(new LstmLayer(embedding_size, l1_size, new Tanh()));
			// nn.add(new DropoutLayer(l1_size));
			nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), labelIndexer);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		}
	}
}
