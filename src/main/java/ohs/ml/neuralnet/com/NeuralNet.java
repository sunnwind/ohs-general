package ohs.ml.neuralnet.com;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.io.FileUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

/**
 * http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
 * 
 * http://cs231n.stanford.edu/syllabus.html
 * 
 * http://cs231n.github.io/
 * 
 * @author ohs
 */
public class NeuralNet extends ArrayList<Layer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8546621903237371147L;

	private Indexer<String> labelIdxer;

	private Vocab vocab;

	/**
	 * 
	 */

	public NeuralNet() {

	}

	public NeuralNet(Indexer<String> labelIdxer, Vocab vocab) {
		this.labelIdxer = labelIdxer;
		this.vocab = vocab;
	}

	public Object backward(DenseMatrix D) {
		Object dY = D;
		Object dX = null;
		for (int i = size() - 1; i >= 0; i--) {
			Layer l = get(i);
			dX = l.backward(dY);
			dY = dX;
		}
		return dX;
	}

	public IntegerArray classify(Object I) {
		DenseMatrix Yh = score(I);
		IntegerArray yh = new IntegerArray(Yh.rowSize());
		for (int i = 0; i < Yh.rowSize(); i++) {
			yh.add(Yh.row(i).argMax());
		}
		return yh;
	}

	public Object forward(Object I) {
		Object X = I;
		Object Y = null;
		for (int i = 0; i < size(); i++) {
			Layer l = get(i);
			Y = l.forward(X);
			X = Y;
		}
		return Y;
	}

	public DenseMatrix getB() {
		List<DenseVector> rows = Generics.newLinkedList();
		for (Layer l : this) {
			if (l.getB() != null) {
				for (DenseVector b : l.getB()) {
					rows.add(b);
				}
			}
		}
		return new DenseMatrix(rows);
	}

	public DenseMatrix getDB() {
		List<DenseVector> rows = Generics.newLinkedList();
		for (Layer l : this) {
			if (l.getDB() != null) {
				for (DenseVector db : l.getDB()) {
					rows.add(db);
				}
			}
		}
		return new DenseMatrix(rows);
	}

	public DenseTensor getDW(boolean no_bias) {
		List<DenseMatrix> ret = Generics.newLinkedList();
		for (Layer l : this) {
			if (l instanceof EmbeddingLayer) {
				EmbeddingLayer n = (EmbeddingLayer) l;
				if (!n.isLearnEmbedding()) {
					continue;
				}
			}

			if (l.getW() != null) {
				ret.add(l.getDW());
			}

			if (!no_bias && l.getDB() != null) {
				ret.add(l.getDB());
			}
		}
		return new DenseTensor(ret);
	}

	public int getInputSize() {
		return get(0).getInputSize();
	}

	public Indexer<String> getLabelIndexer() {
		return labelIdxer;
	}

	public int getOutputSize() {
		return get(size() - 1).getOutputSize();
	}

	public Vocab getVocab() {
		return vocab;
	}

	public DenseTensor getW(boolean no_bias) {
		List<DenseMatrix> ret = Generics.newLinkedList();
		for (Layer l : this) {
			if (l instanceof EmbeddingLayer) {
				EmbeddingLayer n = (EmbeddingLayer) l;
				if (!n.isLearnEmbedding()) {
					continue;
				}
			}

			if (l.getW() != null) {
				ret.add(l.getW());
			}

			if (!no_bias && l.getB() != null) {
				ret.add(l.getB());
			}
		}
		return new DenseTensor(ret);
	}

	public void init() {
		for (Layer l : this) {
			l.init();
		}
	}

	public Layer last() {
		return get(size() - 1);
	}

	public void prepare() {
		for (Layer l : this) {
			l.prepare();
		}
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		labelIdxer = FileUtils.readStringIndexer(ois);
		vocab = new Vocab(ois);

		int size = ois.readInt();

		for (int i = 0; i < size; i++) {
			String name = ois.readUTF();
			Layer l = null;
			if (name.equals(FullyConnectedLayer.class.getName())) {
				l = new FullyConnectedLayer(ois);
			} else if (name.equals(SoftmaxLayer.class.getName())) {
				l = new SoftmaxLayer(ois);
			} else if (name.equals(BatchNormalizationLayer.class.getName())) {
				l = new BatchNormalizationLayer(ois);
			} else if (name.equals(DropoutLayer.class.getName())) {
				l = new DropoutLayer(ois);
			} else if (name.equals(NonlinearityLayer.class.getName())) {
				l = new NonlinearityLayer(ois);
			}
			add(l);
		}
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s]\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public DenseVector score(DenseVector I) {
		DenseMatrix O = (DenseMatrix) forward((Object) new DenseMatrix(new DenseVector[] { I }));
		return O.row(0);
	}

	public DenseMatrix score(Object I) {
		return (DenseMatrix) forward(I);
	}

	public void setIsTesting(boolean is_testing) {
		for (Layer l : this) {
			l.setIsTesting(is_testing);
		}
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, labelIdxer);
		vocab.writeObject(oos);
		oos.writeInt(size());

		for (int i = 0; i < size(); i++) {
			Layer l = get(i);
			oos.writeUTF(l.getClass().getName());
			l.writeObject(oos);
		}
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
