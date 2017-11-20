package ohs.bioasq;

import java.util.List;
import java.util.Map;
import java.util.Set;

import de.bwaldvogel.liblinear.SolverType;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.ml.svm.wrapper.LibLinearTrainer;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class MeshNumberPredictor {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		MeshNumberPredictor gc = new MeshNumberPredictor();
		gc.readData();
		// gc.trainNNs();
		gc.trainLibLinear();
		// gc.testLibLinear();

		System.out.println("process ends.");
	}

	private MeshTree mt;

	private DenseMatrix X;

	private IntegerArray Y;

	public Pair<DenseMatrix, IntegerArray> buildDocumentEmbeddings(IntegerMatrix docData, Map<Integer, DenseVector> docToEmb) {
		List<DenseVector> X = Generics.newArrayList(docData.size());
		IntegerArray Y = new IntegerArray(X.size());
		int label = 0;
		for (IntegerArray docseqs : docData) {
			for (int docseq : docseqs) {
				DenseVector ed = docToEmb.get(docseq);
				X.add(ed);
				Y.add(label);
			}
			label++;
		}
		return Generics.newPair(new DenseMatrix(X), Y);
	}

	public IntegerMatrix getBinaryData(int target, Map<Integer, IntegerArray> topToDoc, boolean balanced) {
		Set<Integer> posSet = Generics.newHashSet();
		Set<Integer> negSet = Generics.newHashSet();

		for (int docseq : topToDoc.get(target)) {
			posSet.add(docseq);
		}

		for (int top_node : topToDoc.keySet()) {
			if (top_node != target) {
				for (int docseq : topToDoc.get(top_node)) {
					if (!posSet.contains(docseq)) {
						negSet.add(docseq);
					}
				}
			}
		}

		IntegerMatrix ret = new IntegerMatrix(2);
		ret.add(new IntegerArray(posSet));
		ret.add(new IntegerArray(negSet));

		if (balanced) {
			int min_size = Integer.MAX_VALUE;

			for (IntegerArray a : ret) {
				min_size = Math.min(min_size, a.size());
			}

			for (int i = 0; i < ret.size(); i++) {
				IntegerArray a = ret.get(i);
				if (a.size() > min_size) {
					ArrayUtils.shuffle(a.values());
					ret.set(i, a.subArray(0, min_size));
				}
			}
		}

		return ret;
	}

	public void readData() throws Exception {
		X = new DenseMatrix(MIRPath.BIOASQ_DIR + "top-doc-embs.ser.gz");
		Y = new IntegerArray(MIRPath.BIOASQ_DIR + "top-mesh-nums.ser.gz");
		mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);
	}

	public void testLibLinear() throws Exception {
		Counter<Integer> labelCnts = Generics.newCounter();

		System.out.println(labelCnts.toString(labelCnts.size()));

		LibLinearWrapper lw = new LibLinearWrapper();
		lw.read(MIRPath.BIOASQ_DIR + String.format("model/ll-mesh_num.ser.gz"));

		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector x = X.row(i);
			int y = Y.get(i);
			double yh = lw.regression(x);
			int yhh = (int) Math.round(yh);
			System.out.println();
		}
	}

	public void trainLibLinear() throws Exception {
		Counter<Integer> labelCnts = Generics.newCounter();

		Indexer<String> labelIdxer = Generics.newIndexer();
		Indexer<String> featIdxer = Generics.newIndexer();

		for (int mesh_num : Y) {
			labelCnts.incrementCount(mesh_num, 1);
			labelIdxer.add(mesh_num + "");
		}

		System.out.println(labelCnts.toString(labelCnts.size()));

		for (int j = 0; j < X.colSize(); j++) {
			featIdxer.add(String.format("feat=%d", j));
		}

		LibLinearTrainer lt = new LibLinearTrainer();
		lt.getParameter().setSolverType(SolverType.L2R_L2LOSS_SVR);

		LibLinearWrapper lw = lt.train(labelIdxer, featIdxer, X, Y);
		lw.write(MIRPath.BIOASQ_DIR + String.format("model/ll-mesh_num.ser.gz"));
	}

	public void trainNNs() throws Exception {
		Counter<Integer> labelCnts = Generics.newCounter();

		for (int mesh_num : Y) {
			labelCnts.incrementCount(mesh_num, 1);
		}

		IntegerArray tops = new IntegerArray(labelCnts.getSortedKeys());
		tops.reverse();

		MeshTree mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);

		System.out.println(labelCnts.toString(labelCnts.size()));

		int input_size = X.colSize();
		int l1_size = 100;
		int l2_size = 50;
		int output_size = labelCnts.size();

		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(input_size);
		param.setHiddenSize(50);
		param.setOutputSize(output_size);
		param.setBatchSize(50);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(4);

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(input_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(l1_size, new Tanh()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(l2_size, new Tanh()));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepareTraining();
		nn.initWeights();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.size(), null);
		trainer.train(X, Y, null, null, 10000);
		trainer.finish();

		nn.writeObject(MIRPath.BIOASQ_DIR + String.format("model/nn-mesh-num.ser.gz"));
	}

}
