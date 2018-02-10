package ohs.ml.neuralnet.com;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

/**
 * 
 * http://cs231n.github.io/neural-networks-2/#init
 * 
 * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 * 
 * http://pythonkim.tistory.com/41
 * 
 * @author ohs
 */
public class ParameterInitializer {

	public static enum Type {
		BASIC, XAVIER, HE;
	}

	private static void basic(DenseMatrix W) {
		int input_size = W.rowSize();
		double bound = 1d / Math.sqrt(input_size);
		VectorMath.random(-bound, bound, W);
	}

	private Type type;

	public ParameterInitializer() {
		this(Type.HE);
	}

	public ParameterInitializer(Type type) {
		this.type = type;
	}

	private void He(DenseMatrix W) {
		VectorMath.randomn(0, 1, W);

		int input_size = W.rowSize();
		double scale = 1d / Math.sqrt(input_size / 2);
		W.multiply(scale);

		VectorMath.clip(W, -1, 1, W);
	}

	public void init(DenseMatrix W) {
		if (type == Type.BASIC) {
			basic(W);
		} else if (type == Type.XAVIER) {
			Xavier(W);
		} else if (type == Type.HE) {
			He(W);
		}
	}

	private void Xavier(DenseMatrix W) {
		VectorMath.randomn(0, 1, W);

		int input_size = W.rowSize();
		double scale = 1d / Math.sqrt(input_size);
		W.multiply(scale);

		VectorMath.clip(W, -1, 1, W);
	}

}
