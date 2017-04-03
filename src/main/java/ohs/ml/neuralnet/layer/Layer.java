package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import ohs.matrix.DenseMatrix;

public abstract class Layer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1162846301055604737L;

	protected boolean is_testing = false;

	public Object backward(Object I) {
		return null;
	}

	public Object forward(Object I) {
		return null;
	}

	public DenseMatrix getB() {
		return null;
	}

	public DenseMatrix getDB() {
		return null;
	}

	public DenseMatrix getDW() {
		return null;
	}

	public int getInputSize() {
		return 0;
	}

	public int getOutputSize() {
		return 0;
	}

	public DenseMatrix getW() {
		return null;
	}

	public void init() {
	}

	public boolean isTesting() {
		return is_testing;
	}

	public void prepare() {

	}

	public void readObject(ObjectInputStream ois) throws Exception {

	}

	public void setIsTesting(boolean is_testing) {
		this.is_testing = is_testing;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {

	}

}
