package ohs.ml.neuralnet.apps;

import java.util.Set;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.DataReader;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.com.TaskType;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class MNISTApp {
	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		run();
		System.out.println("process ends.");
	}

	public static void run() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setInputSize(100);
		nnp.setHiddenSize(50);
		nnp.setOutputSize(10);
		nnp.setBatchSize(50);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setGradientClipCutoff(20);
		nnp.setOptimizerType(OptimizerType.ADAM);

		DenseTensor X = new DenseTensor();
		DenseTensor Y = new DenseTensor();

		DenseTensor Xt = new DenseTensor();
		DenseTensor Yt = new DenseTensor();

		Indexer<String> labelIdxer = Generics.newIndexer();
		Vocab vocab = new Vocab();

		{

			Pair<SparseMatrix, IntegerArray> train = DataReader.readFromSvmFormat("../../data/ml_data/mnist.txt");
			Pair<SparseMatrix, IntegerArray> test = DataReader.readFromSvmFormat("../../data/ml_data/mnist.t.txt");

			DenseMatrix _X = train.getFirst().toDenseMatrix();
			IntegerArray _Y = train.getSecond();

			DenseMatrix _Xt = test.getFirst().toDenseMatrix(test.getFirst().rowSize(), _X.colSize());
			IntegerArray _Yt = test.getSecond();

			Set<String> labels = Generics.newTreeSet();
			for (int y : _Y) {
				labels.add(y + "");
			}

			labelIdxer = Generics.newIndexer(labels);

			for (int i = 0; i < _X.colSize(); i++) {
				vocab.add(i + "");
			}

			X.ensureCapacity(_Y.size());
			Y.ensureCapacity(_Yt.size());

			for (int i = 0; i < _X.rowSize(); i++) {
				X.add(_X.row(i).toDenseMatrix());
				Y.add(new DenseVector(new double[] { _Y.get(i) }).toDenseMatrix());
			}

			Xt.ensureCapacity(_Xt.rowSize());
			Yt.ensureCapacity(_Yt.size());

			for (int i = 0; i < _Xt.rowSize(); i++) {
				Xt.add(_Xt.row(i).toDenseMatrix());
				Yt.add(new DenseVector(new double[] { _Yt.get(i) }).toDenseMatrix());
			}
		}

		int vocab_size = vocab.size();
		int l1_size = 100;
		int l2_size = 25;
		int output_size = labelIdxer.size();

		NeuralNet nn = new NeuralNet(labelIdxer, null, TaskType.CLASSIFICATION);

		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.createGradientHolders();
		nn.initWeights(new ParameterInitializer());

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}
}
