package ohs.ml.neuralnet.com;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;

public class ViterbiDecoder {

	/**
	 * emission probabilities
	 */
	private DenseMatrix E;

	/**
	 * start probabilities
	 */
	private DenseVector S;

	/**
	 * number of states
	 */
	private int state_size;

	private Indexer<String> stateIndexer;

	/**
	 * transition probabilities
	 */
	private DenseMatrix T;

	/**
	 * number of unique observations in vocabulary
	 */
	private int vocab_size;

	private Indexer<String> wordIndexer;

	public ViterbiDecoder(DenseVector S, DenseMatrix T, DenseMatrix E) {
		this.T = T;
		this.E = E;
		this.S = S;

		state_size = E.rowSize();
		vocab_size = E.colSize();
	}

	public ViterbiDecoder(Indexer<String> stateIndexer, Indexer<String> wordIndexer) {
		this.stateIndexer = stateIndexer;
		this.wordIndexer = wordIndexer;

		state_size = stateIndexer.size();
		vocab_size = wordIndexer.size();

		S = new DenseVector(state_size);
		T = new DenseMatrix(state_size, state_size);
		E = new DenseMatrix(state_size, vocab_size);

		VectorMath.random(0, 1, S);
		VectorMath.random(0, 1, T);
		VectorMath.random(0, 1, E);

		S.normalize();
		T.normalizeRows();
		E.normalizeRows();
	}

	public ViterbiDecoder(int state_size, int vocab_size) {
		this.state_size = state_size;
		this.vocab_size = vocab_size;

		S = new DenseVector(state_size);
		T = new DenseMatrix(state_size, state_size);
		E = new DenseMatrix(state_size, vocab_size);

		VectorMath.random(0, 1, S);
		VectorMath.random(0, 1, T);
		VectorMath.random(0, 1, E);

		S.normalize();
		T.normalizeRows();
		E.normalizeRows();
	}

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

	public DenseVector decode(DenseVector O) {
		int seq_len = O.size();

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

		DenseVector bestPath = new DenseVector(seq_len);

		VectorUtils.copy(F.row(F.rowSize() - 1), tmp);

		int q = tmp.argMax();

		for (int t = seq_len - 1; t >= 0; t--) {
			bestPath.set(t, q);
			q = (int) backPointers.value(t, q);
		}

		return bestPath;
	}

	public IntegerArray decode(IntegerArray O) {
		int seq_len = O.size();

		DenseMatrix F = new DenseMatrix(seq_len, state_size);
		DenseMatrix backPointers = new DenseMatrix(seq_len, state_size);

		VectorMath.multiply(S, E.row(O.get(0)), F.row(0));

		DenseVector tmp = new DenseVector(state_size);

		for (int t = 1; t < seq_len; t++) {

			for (int s = 0; s < state_size; s++) {
				tmp.setAll(0);

				for (int s_prev = 0; s_prev < state_size; s_prev++) {
					tmp.set(s_prev, F.value(t - 1, s_prev) * T.value(s_prev, s));
				}

				int s_max = tmp.argMax();

				F.set(t, s, tmp.value(s_max) * E.value(s, O.get(t)));

				backPointers.set(t, s, s_max);
			}
		}

		IntegerArray bestPath = new IntegerArray(ArrayUtils.range(seq_len));

		VectorUtils.copy(F.row(F.rowSize() - 1), tmp);

		int q = tmp.argMax();

		for (int t = seq_len - 1; t >= 0; t--) {
			bestPath.set(t, q);
			q = (int) backPointers.value(t, q);
		}

		return bestPath;
	}

	public DenseMatrix getEmissionProbs() {
		return E;
	}

	public DenseVector getStartProbs() {
		return S;
	}

	public int getStateSize() {
		return state_size;
	}

	public DenseMatrix getTransitionProbs() {
		return T;
	}

	public int getVocabSize() {
		return vocab_size;
	}

	public void print() {
	}

	public void read(ObjectInputStream ois) throws Exception {
		stateIndexer = FileUtils.readStringIndexer(ois);
		wordIndexer = FileUtils.readStringIndexer(ois);

		S = new DenseVector(ois);
		T = new DenseMatrix(ois);
		E = new DenseMatrix(ois);

		state_size = stateIndexer.size();
		vocab_size = wordIndexer.size();
	}

	public void read(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		read(ois);
		ois.close();
	}

	public void setEmissionProbs(DenseMatrix E) {
		this.E = E;
	}

	public void setStartProbs(DenseVector phi) {
		this.S = phi;
	}

	public void setTransitionProbs(DenseMatrix T) {
		this.T = T;
	}

	public String[] tag(String[] words) {
		int[] ws = wordIndexer.indexesOf(words).values();
		IntegerArray sts = decode(new IntegerArray(ws));
		String[] ret = stateIndexer.getObjects(sts.clone().values());
		return ret;
	}

	public void write(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, stateIndexer);
		FileUtils.writeStringIndexer(oos, wordIndexer);

		S.writeObject(oos);
		T.writeObject(oos);
		E.writeObject(oos);
	}

	public void write(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		write(oos);
		oos.close();
	}

}
