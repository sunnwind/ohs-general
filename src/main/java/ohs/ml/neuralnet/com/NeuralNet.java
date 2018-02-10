package ohs.ml.neuralnet.com;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.LinkedList;

import org.openxmlformats.schemas.drawingml.x2006.spreadsheetDrawing.STEditAs;

import ohs.io.FileUtils;
import ohs.math.ArrayChecker;
import ohs.math.VectorChecker;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.hmm.ViterbiDecoder;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
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

	public static DenseVector decode(DenseVector O, DenseVector S, DenseMatrix T, DenseMatrix E) {
		int seq_len = O.size();
		int state_size = T.rowSize();

		DenseMatrix F = new DenseMatrix(seq_len, state_size);
		DenseMatrix backPointers = new DenseMatrix(seq_len, state_size);

		VectorMath.multiply(S, E.row((int) O.value(0)), F.row(0));

		DenseVector tmp = new DenseVector(state_size);

		for (int t = 1; t < seq_len; t++) {

			for (int s = 0; s < state_size; s++) {
				tmp.setAll(0);

				for (int s_prev = 0; s_prev < state_size; s_prev++) {
					tmp.set(s_prev, F.value(t - 1, s_prev) * T.value(s_prev, s));
				}

				int s_max = tmp.argMax();

				F.set(t, s, tmp.value(s_max) * E.value(s, (int) O.value(t)));

				backPointers.set(t, s, s_max);
			}
		}

		DenseVector ret = new DenseVector(seq_len);

		VectorUtils.copy(F.row(F.rowSize() - 1), tmp);

		int q = tmp.argMax();

		for (int t = seq_len - 1; t >= 0; t--) {
			ret.set(t, q);
			q = (int) backPointers.value(t, q);
		}

		return ret;
	}

	private Indexer<String> labelIdxer = Generics.newIndexer();

	private Vocab vocab = new Vocab();

	private TaskType tt = TaskType.CLASSIFICATION;

	private boolean use_viterbi = true;

	private boolean is_training = false;

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

	public Layer getLast() {
		return get(size() - 1);
	}

	public DenseTensor classify(Object I) {
		DenseTensor Yh = (DenseTensor) forward(I);
		DenseTensor ret = new DenseTensor();
		ret.ensureCapacity(Yh.size());

		for (int i = 0; i < Yh.size(); i++) {
			DenseMatrix Yhm = Yh.get(i);
			DenseMatrix P1 = new DenseMatrix(Yhm.rowSize(), 1);

			for (int j = 0; j < Yhm.rowSize(); j++) {
				P1.row(j).add(0, Yhm.row(j).argMax());
			}
			ret.add(P1);
		}
		return ret;
	}

	public DenseTensor classify(Object I, boolean use_viterbi) {
		DenseTensor Yh = (DenseTensor) forward(I);
		DenseTensor ret = new DenseTensor();
		ret.ensureCapacity(Yh.size());

		for (int i = 0; i < Yh.size(); i++) {
			DenseMatrix Yhm = Yh.get(i);
			DenseMatrix P = new DenseMatrix(Yhm.rowSize(), 1);

			for (int j = 0; j < Yhm.rowSize(); j++) {
				P.row(j).add(0, Yhm.row(j).argMax());
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

	public void createGradientHolders() {
		for (Layer l : this) {
			l.createGradientHolders();
			l.setIsTraining(true);
		}
	}

	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = null;

		for (int i = 0; i < size(); i++) {
			Layer l = get(i);
			Y = (DenseTensor) l.forward(X);
			X = Y;
		}

		if (!is_training) {
			Y = Y.copy(false);
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
				} else if (l instanceof BidirectionalRecurrentLayer) {
					BidirectionalRecurrentLayer n = (BidirectionalRecurrentLayer) l;
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
				} else if (l instanceof BidirectionalRecurrentLayer) {
					BidirectionalRecurrentLayer n = (BidirectionalRecurrentLayer) l;
					W = n.getW();
					B = n.getB();
				}
			}

			if (W != null && W.size() > 0) {
				ret.addAll(W);
			}

			if (!no_bias && B != null && B.size() > 0) {
				ret.addAll(B);
			}
		}
		ret.unwrapValues();
		return ret;
	}

	public void initWeights(ParameterInitializer pi) {
		for (Layer l : this) {
			l.initWeights(pi);
		}
	}

	public Layer last() {
		return get(size() - 1);
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

	public void setIsTraining(boolean is_training) {
		this.is_training = is_training;

		for (Layer l : this) {
			l.setIsTraining(is_training);
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