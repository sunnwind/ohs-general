package ohs.bioasq;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
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

public class TopMeshClassifier {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		TopMeshClassifier gc = new TopMeshClassifier();
		gc.readData();
		gc.trainNNs();
		// gc.trainLibLinear();

		System.out.println("process ends.");
	}

	private Map<Integer, DenseVector> docToEmb;

	private Map<Integer, IntegerArray> topToDoc;

	private SparseVector topDocCnts;

	private MeshTree mt;

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

	public Pair<DenseMatrix, IntegerArray> getDocumentEmbeddings(IntegerMatrix docData, Map<Integer, DenseVector> docToEmb) {
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

	public void readData() throws Exception {
		docToEmb = Generics.newHashMap();

		DenseMatrix ED = new DenseMatrix(MIRPath.BIOASQ_DIR + "top-doc-embs.ser.gz");
		IntegerArray docseqs = new IntegerArray(MIRPath.BIOASQ_DIR + "top-docseqs.ser.gz");

		for (int i = 0; i < docseqs.size(); i++) {
			int docseq = docseqs.get(i);
			DenseVector ed = ED.row(i);
			docToEmb.put(docseq, ed);
		}

		topToDoc = Generics.newHashMap();

		Counter<Integer> c = Generics.newCounter();

		for (int top : topToDoc.keySet()) {
			c.setCount(top, topToDoc.get(top).size());
		}

		for (File file : FileUtils.getFilesUnder(MIRPath.BIOASQ_DIR + "doc-class/top-mesh/")) {
			String fileName = file.getName();
			int idx = fileName.indexOf(".");
			fileName = fileName.substring(0, idx);
			int top = Integer.parseInt(fileName);
			topToDoc.put(top, new IntegerArray(file.getPath()));
			c.incrementCount(top, topToDoc.get(top).size());
		}

		topDocCnts = new SparseVector(c);

		mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);
	}

	public void trainLibLinear() throws Exception {

		topDocCnts.sortValues(false);

		for (int i = 0; i < topDocCnts.size(); i++) {
			int top = topDocCnts.indexAt(i);
			Pair<DenseMatrix, IntegerArray> p = getDocumentEmbeddings(getBinaryData(top, topToDoc, true), docToEmb);
			DenseMatrix X = p.getFirst();
			IntegerArray Y = p.getSecond();

			int[] locs = ArrayUtils.range(Y.size());
			ArrayUtils.shuffle(locs);

			int mid = X.rowSize() / 2;

			for (int m = 0; m < mid; m++) {
				int n = locs[m];
				X.swapRows(m, n);
				Y.swap(m, n);
			}

			Counter<String> labels = Generics.newCounter();

			String topMesh = mt.getName(top);

			for (int y : Y) {
				if (y == 0) {
					labels.incrementCount(topMesh, 1);
				} else {
					labels.incrementCount("Other", 1);
				}
			}

			Indexer<String> labelIdxer = Generics.newIndexer();
			Indexer<String> featIdxer = Generics.newIndexer();

			labelIdxer.add(topMesh);
			labelIdxer.add("Other");

			for (int j = 0; j < X.colSize(); j++) {
				featIdxer.add(String.format("feat=%d", j));
			}

			LibLinearTrainer lt = new LibLinearTrainer();
			LibLinearWrapper lw = lt.train(labelIdxer, featIdxer, X, Y);
			lw.write(MIRPath.BIOASQ_DIR + String.format("liblinear/global-%d.ser.gz", top));
		}
	}

	public void trainNNs() throws Exception {

		for (int i = 0; i < topDocCnts.size(); i++) {
			int top = topDocCnts.indexAt(i);
			Pair<DenseMatrix, IntegerArray> p = getDocumentEmbeddings(getBinaryData(top, topToDoc, true), docToEmb);
			DenseMatrix X = p.getFirst();
			IntegerArray Y = p.getSecond();

			Counter<String> labels = Generics.newCounter();

			String topName = mt.getName(top);

			for (int y : Y) {
				if (y == 0) {
					labels.incrementCount(topName, 1);
				} else {
					labels.incrementCount("Other", 1);
				}
			}

			System.out.println(labels.toString());

			int input_size = X.colSize();
			int l1_size = 200;
			int l2_size = 50;
			int output_size = labels.size();

			NeuralNetParams param = new NeuralNetParams();
			param.setInputSize(input_size);
			param.setHiddenSize(50);
			param.setOutputSize(output_size);
			param.setBatchSize(50);
			param.setLearnRate(0.001);
			param.setRegLambda(0.001);
			param.setThreadSize(20);

			NeuralNet nn = new NeuralNet();
			nn.add(new FullyConnectedLayer(input_size, l1_size));
			nn.add(new BatchNormalizationLayer(l1_size));
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
			trainer.train(X, Y, null, null, 2000);
			trainer.finish();

			nn.writeObject(MIRPath.BIOASQ_DIR + String.format("nn/top-mesh-%d.ser.gz", top));
		}
	}

}
