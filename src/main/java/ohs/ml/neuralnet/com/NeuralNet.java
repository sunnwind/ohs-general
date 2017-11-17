package ohs.ml.neuralnet.com;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.io.FileUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.layer.BidirectionalRNN;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
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

	private Indexer<String> labelIdxer = Generics.newIndexer();

	private Vocab vocab = new Vocab();

	private TaskType tt;

	/**
	 * 
	 */

	public NeuralNet() {

	}

	public NeuralNet(Indexer<String> labelIdxer, Vocab vocab, TaskType tt) {
		this.labelIdxer = labelIdxer;
		this.vocab = vocab;
		this.tt = tt;
	}

	public NeuralNet(String fileName) throws Exception {
		readObject(fileName);
	}

	public Object backward(DenseTensor D) {
		Object dY = D;
		Object dX = null;
		for (int i = size() - 1; i >= 0; i--) {
			Layer l = get(i);
			dX = l.backward(dY);
			dY = dX;
		}
		return dX;
	}

	public DenseMatrix classify(Object I) {
		DenseTensor Yh = (DenseTensor) score(I);
		DenseMatrix ret = new DenseMatrix();
		ret.ensureCapacity(Yh.size());

		for (int i = 0; i < Yh.size(); i++) {
			DenseMatrix Yhm = Yh.get(i);
			DenseVector P = new DenseVector(Yhm.rowSize());

			for (int j = 0; j < Yhm.rowSize(); j++) {
				P.add(j, Yhm.row(j).argMax());
			}
			ret.add(P);
		}
		return ret;
	}

	public NeuralNet copy() {
		NeuralNet ret = new NeuralNet(labelIdxer, vocab, tt);
		ret.ensureCapacity(size());

		for (Layer l : this) {
			ret.add(l.copy());
		}
		return ret;
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

	public DenseTensor getDW(boolean no_bias) {
		DenseTensor ret = new DenseTensor();
		for (Layer l : this) {
			DenseTensor dW = l.getDW();
			DenseTensor dB = l.getDB();

			if (l instanceof EmbeddingLayer) {
				EmbeddingLayer n = (EmbeddingLayer) l;
				if (n.isLearnEmbedding()) {
					dW = n.getDW();
					dB = n.getDB();
				}
			} else if (l instanceof RecurrentLayer) {
				if (l instanceof RnnLayer) {
					RnnLayer n = (RnnLayer) l;
					dW = n.getDW();
					dB = n.getDB();
				} else if (l instanceof LstmLayer) {
					LstmLayer n = (LstmLayer) l;
					dW = n.getDW();
					dB = n.getDB();
				} else if (l instanceof BidirectionalRNN) {
					BidirectionalRNN n = (BidirectionalRNN) l;
					dW = n.getDW();
					dB = n.getDB();
				}
			}

			if (dW != null) {
				ret.addAll(dW);
			}

			if (!no_bias && dB != null) {
				ret.addAll(dB);
			}
		}
		return ret;
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

	public TaskType getTaskType() {
		return tt;
	}

	public Vocab getVocab() {
		return vocab;
	}

	public DenseTensor getW(boolean no_bias) {
		DenseTensor ret = new DenseTensor();
		for (Layer l : this) {
			DenseTensor W = l.getW();
			DenseTensor B = l.getB();

			if (l instanceof EmbeddingLayer) {
				EmbeddingLayer n = (EmbeddingLayer) l;
				if (n.isLearnEmbedding()) {
					W = n.getW();
					B = n.getB();
				}
			} else if (l instanceof RecurrentLayer) {
				if (l instanceof RnnLayer) {
					RnnLayer n = (RnnLayer) l;
					W = n.getW();
					B = n.getB();
				} else if (l instanceof LstmLayer) {
					LstmLayer n = (LstmLayer) l;
					W = n.getW();
					B = n.getB();
				} else if (l instanceof BidirectionalRNN) {
					BidirectionalRNN n = (BidirectionalRNN) l;
					W = n.getW();
					B = n.getB();
				}
			}

			if (W != null) {
				ret.addAll(W);
			}

			if (!no_bias && B != null) {
				ret.addAll(B);
			}
		}
		ret.unwrapValues();
		return ret;
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
		this.clear();

		tt = TaskType.values()[ois.readInt()];
		labelIdxer = FileUtils.readStringIndexer(ois);
		vocab = new Vocab(ois);

		int size = ois.readInt();

		for (int i = 0; i < size; i++) {
			String name = ois.readUTF();
			Class c = Class.forName(name);
			Layer l = (Layer) c.getDeclaredConstructor().newInstance();
			l.readObject(ois);
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

	public Object score(Object I) {
		return forward(I);
	}

	public void setIsTesting(boolean is_testing) {
		for (Layer l : this) {
			l.setIsTesting(is_testing);
		}
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(tt.ordinal());
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
		oos.flush();
		oos.close();
	}

}