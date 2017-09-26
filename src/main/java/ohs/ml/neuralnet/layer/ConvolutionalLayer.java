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

	/**
	 * 
	 */
	private static final long serialVersionUID = 3645948244034058451L;

	private ConvLayer conv;

	private NonlinearityLayer nl;

	private int num_filters;

	private int output_size;

	private int emb_size;

	private MaxPoolingLayer pl;

	private DenseVector y;

	public ConvolutionalLayer(ConvLayer conv, NonlinearityLayer nl, MaxPoolingLayer pl) {
		this.conv = conv;
		this.nl = nl;
		this.pl = pl;
	}

	public ConvolutionalLayer(int emb_size, int[] window_sizes, int num_filters) {
		this.emb_size = emb_size;
		this.num_filters = num_filters;
		this.output_size = window_sizes.length * num_filters;

		conv = new ConvLayer(emb_size, 3, num_filters);

		nl = new NonlinearityLayer(output_size, new ReLU());

		pl = new MaxPoolingLayer(num_filters);

		y = new DenseVector(output_size);
	}

	@Override
	public Object backward(Object I) {
		return null;
	}

	@Override
	public Object forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;

		DenseMatrix T1 = (DenseMatrix) conv.forward(X);
		DenseMatrix T2 = (DenseMatrix) nl.forward(T1);
		DenseMatrix T3 = (DenseMatrix) pl.forward(T2);

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
		return emb_size;
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
