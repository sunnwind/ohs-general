package ohs.ml.neuralnet.com;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;

/**
 * 
 * http://cs231n.github.io/neural-networks-2/#init
 * 
 * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 * 
 * @author ohs
 */
public class ParameterInitializer {

	public static void init1(DenseMatrix W) {
		int input_size = W.rowSize();
		double bound = 1d / Math.sqrt(input_size);
		VectorMath.random(-bound, bound, W);
	}

	public static void init2(DenseMatrix W) {
		int input_size = W.rowSize();
		int output_size = W.colSize();
		double bound = Math.sqrt(6) / Math.sqrt(input_size + output_size);
		VectorMath.random(-bound, bound, W);
	}

	public static void init3(DenseMatrix W) {
		int input_size = W.rowSize();
		double bound = Math.sqrt(2 / input_size);
		VectorMath.random(-bound, bound, W);
	}

}
