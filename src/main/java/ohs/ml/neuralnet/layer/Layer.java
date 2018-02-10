package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import ohs.matrix.DenseTensor;
import ohs.ml.neuralnet.com.ParameterInitializer;

public abstract class Layer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1162846301055604737L;

	protected boolean is_training = false;

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

	public void initWeights(ParameterInitializer pi) {
	}

	public boolean isTraining() {
		return is_training;
	}

	public void createGradientHolders() {
		is_training = true;
	}

	public void readObject(ObjectInputStream ois) throws Exception {

	}

	public void setIsTraining(boolean is_training) {
		this.is_training = is_training;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {

	}

}
