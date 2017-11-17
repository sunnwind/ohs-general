package ohs.ir.search.app;

import ohs.ir.medical.general.MIRPath;
import ohs.matrix.DenseMatrix;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class L2RReranker {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		train();

		System.out.println("process ends.");
	}

	public static void train() throws Exception {

		IntegerArray Y = new IntegerArray();
		DenseMatrix X = new DenseMatrix();

		IntegerArray Yt = new IntegerArray();
		DenseMatrix Xt = new DenseMatrix();

		Y.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2015_label.ser.gz");
		X.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2015_data.ser.gz");

		Yt.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2014_label.ser.gz");
		Xt.readObject(MIRPath.TREC_CDS_2014_DIR + "l2r_2014_data.ser.gz");

		System.out.println(X.size());
		System.out.println(Xt.size());

		for (int i = 0; i < Y.size(); i++) {
			int relevance = Y.get(i);
			int label = 0;
			if (relevance > 0) {
				label = 1;
			}
			Y.set(i, label);
		}

		for (int i = 0; i < Yt.size(); i++) {
			int relevance = Yt.get(i);
			int label = 0;
			if (relevance > 0) {
				label = 1;
			}
			Yt.set(i, label);
		}

		CounterMap<String, Integer> cm = Generics.newCounterMap();

		for (int i = 0; i < Y.size(); i++) {
			int label = Y.get(i);
			cm.incrementCount("Train", label, 1);
			// if (label > 0) {
			// label = 1;
			// }

			Y.set(i, label);
		}

		for (int i = 0; i < Yt.size(); i++) {
			int label = Yt.get(i);
			cm.incrementCount("Test", label, 1);
			// if (label > 0) {
			// label = 1;
			// }

			Yt.set(i, label);
		}

		System.out.println(cm);

		NeuralNetParams param = new NeuralNetParams();
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(2);

		Counter<Integer> labels = Generics.newCounter();
		for (int y : Y) {
			labels.incrementCount(y, 1);
		}

		int input_size = X.colSize();
		int l1_size = 300;
		int l2_size = 100;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(input_size, l1_size));
		nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));

		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
		trainer.train(X, Y, Xt, Yt, 100);
		trainer.finish();

		nn.writeObject(MIRPath.TREC_CDS_2014_DIR + "reranker.ser.gz");

		IntegerArray Yth = nn.classify(Xt);

		// FileUtils.writeStringCollection(KPPath.KYP_DIR + "phrs_test_S_res.txt.gz", lines);
	}

}
