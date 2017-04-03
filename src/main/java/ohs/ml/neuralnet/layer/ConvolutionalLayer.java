package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.utils.Generics;

/**
 * http://cthorey.github.io./backprop_conv/
 * 
 * http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 * 
 * @author ohs
 *
 */
public class ConvolutionalLayer extends Layer {

	private List<NeuralNet> conns;

	private int output_size;

	private DenseVector y;

	private int num_filters;

	public ConvolutionalLayer(int embedding_size, int[] window_sizes, int num_filters) {
		this.num_filters = num_filters;
		this.output_size = window_sizes.length * num_filters;

		conns = Generics.newArrayList(window_sizes.length);

		// for (int i = 0; i < window_sizes.length; i++) {
		// NeuralNet nn = new NeuralNet();
		// nn.add(new ConvLayer(embedding_size, window_sizes[i], num_filters));
		// nn.add(new NonlinearityLayer(0, new ReLU()));
		// nn.add(new MaxPoolingLayer(num_filters));
		// conns.add(nn);
		// }

		y = new DenseVector(output_size);
	}

	@Override
	public Object backward(Object I) {
		return null;
	}

	@Override
	public Object forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;
		List<DenseVector> ys = Generics.newArrayList();
		for (int i = 0; i < conns.size(); i++) {
			NeuralNet nn = conns.get(i);
			DenseVector y = (DenseVector) nn.forward(X);
			ys.add(y);
		}

		VectorUtils.copyRows(new DenseMatrix(ys), 0, ys.size(), y);
		return y.toDenseMatrix();
	}

	@Override
	public DenseMatrix getB() {
		return null;
	}

	@Override
	public DenseMatrix getDB() {
		return null;
	}

	@Override
	public DenseMatrix getDW() {
		return null;
	}

	@Override
	public int getInputSize() {
		return num_filters;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	@Override
	public DenseMatrix getW() {
		return null;
	}

	@Override
	public void init() {
		for (NeuralNet nn : conns) {
			nn.init();
		}
	}

}
