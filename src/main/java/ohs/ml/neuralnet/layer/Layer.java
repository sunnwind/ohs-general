package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;

public abstract class Layer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1162846301055604737L;

	protected boolean is_testing = false;

	public Layer() {

	}

	public Object backward(Object I) {
		return I;
	}

	public Layer copy() {
		throw new NullPointerException("copy is not defined.");
	}

	public Object forward(Object I) {
		return I;
	}

	public DenseTensor getB() {
		return null;
	}

	public DenseTensor getDB() {
		return null;
	}

	public DenseTensor getDW() {
		return null;
	}

	public int getInputSize() {
		return 0;
	}

	public int getOutputSize() {
		return 0;
	}

	public DenseTensor getW() {
		return null;
	}

	public void initWeights() {
	}

	public boolean isTesting() {
		return is_testing;
	}

	public void prepareTraining() {

	}

	public void readObject(ObjectInputStream ois) throws Exception {

	}

	public void setIsTesting(boolean is_testing) {
		this.is_testing = is_testing;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {

	}

}
