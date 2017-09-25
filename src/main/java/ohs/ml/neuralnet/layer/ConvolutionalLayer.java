package ohs.ml.neuralnet.layer;

import java.util.List;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.NeuralNet;
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

	private Layer conv;

	private Layer nl;

	private Layer pl;

	private int output_size;

	private DenseVector y;

	private int filter_size;

	public ConvolutionalLayer(Layer conv, Layer nl, Layer pl) {
		this.conv = conv;
		this.nl = nl;
		this.pl = pl;
	}

	public ConvolutionalLayer(int emb_size, int[] window_sizes, int filter_size) {
		this.filter_size = filter_size;
		this.output_size = window_sizes.length * filter_size;

		conv = new ConvLayer(emb_size, 3, filter_size);

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
		return filter_size;
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
	}

}
