package ohs.ml.hmm;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;

public class HMM {

	/**
	 * start probabilities
	 */
	private DenseVector phi;

	/**
	 * transition probabilities
	 */
	private DenseMatrix a;

	/**
	 * emission probabilities
	 */
	private DenseMatrix b;

	/**
	 * number of states
	 */
	private int N;

	/**
	 * number of unique observations in vocabulary
	 */
	private int V;

	private SparseMatrix bigramProbs = new SparseMatrix(new int[0], new SparseVector[0]);

	private Indexer<String> stateIndexer;

	private Indexer<String> wordIndexer;

	public HMM(Indexer<String> stateIndexer, Indexer<String> wordIndexer) {
		this.stateIndexer = stateIndexer;
		this.wordIndexer = wordIndexer;

		N = stateIndexer.size();
		V = wordIndexer.size();

		phi = new DenseVector(N);
		a = new DenseMatrix(N, N);
		b = new DenseMatrix(N, V);

		VectorMath.random(0, 1, phi);
		VectorMath.random(0, 1, a);
		VectorMath.random(0, 1, b);

		phi.normalize();
		a.normalizeRows();
		b.normalizeRows();
	}

	public HMM(DenseVector phi, DenseMatrix a, DenseMatrix b) {
		this.a = a;
		this.b = b;
		this.phi = phi;

		N = b.rowSize();
		V = b.colSize();
	}

	public DenseMatrix getA() {
		return a;
	}

	public DenseMatrix getB() {
		return b;
	}

	public int getN() {
		return N;
	}

	public DenseVector getPhi() {
		return phi;
	}

	public int getV() {
		return V;
	}

	public void print() {
	}

	public void read(ObjectInputStream ois) throws Exception {
		stateIndexer = FileUtils.readStringIndexer(ois);
		wordIndexer = FileUtils.readStringIndexer(ois);

		phi = new DenseVector(ois);
		a = new DenseMatrix(ois);
		b = new DenseMatrix(ois);

		N = stateIndexer.size();
		V = wordIndexer.size();
	}

	public void read(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		read(ois);
		ois.close();
	}

	public void setA(DenseMatrix a) {
		this.a = a;
	}

	public void setB(DenseMatrix b) {
		this.b = b;
	}

	public void setParams(DenseVector phi, DenseMatrix a, DenseMatrix b) {
		this.phi = phi;
		this.a = a;
		this.b = b;
	}

	public void setPhi(DenseVector phi) {
		this.phi = phi;
	}

	public String[] tag(String[] words) {
		int[] ws = wordIndexer.indexesOf(words).values();
		IntegerArray sts = viterbi(new IntegerArray(ws));
		String[] ret = stateIndexer.getObjects(sts.clone().values());
		return ret;
	}

	public IntegerArray viterbi(IntegerArray obs) {
		int T = obs.size();
		DenseMatrix fwd = new DenseMatrix(N, T);
		int[][] backPointers = ArrayMath.matrixInt(N, T);

		for (int i = 0; i < N; i++) {
			fwd.set(i, 0, phi.value(i) * b.value(i, obs.get(0)));
		}

		DenseVector tmp = new DenseVector(N);

		for (int t = 1; t < T; t++) {
			for (int j = 0; j < N; j++) {
				tmp.setAll(0);
				for (int i = 0; i < N; i++) {
					tmp.set(i, fwd.value(i, t - 1) * a.value(i, j));
				}
				int k = tmp.argMax();
				fwd.set(j, t, tmp.value(k) * b.value(j, obs.get(t)));
				backPointers[j][t] = k;
			}
		}

		IntegerArray bestPath = new IntegerArray(ArrayUtils.range(T));

		VectorUtils.copyColumn(fwd, T - 1, tmp);

		int q = tmp.argMax();

		for (int t = T - 1; t >= 0; t--) {
			bestPath.set(t, q);
			q = backPointers[q][t];
		}
		return bestPath;
	}

	public void write(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, stateIndexer);
		FileUtils.writeStringIndexer(oos, wordIndexer);

		phi.writeObject(oos);
		a.writeObject(oos);
		b.writeObject(oos);
	}

	public void write(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		write(oos);
		oos.close();
	}

}
