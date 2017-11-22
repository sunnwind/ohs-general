package ohs.ml.neuralnet.layer;

import ohs.matrix.DenseTensor;
import ohs.ml.neuralnet.nonlinearity.ReLU;

/**
 * http://cthorey.github.io./backprop_conv/
 * 
 * http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
 * 
 * @author ohs
 *
 */
public class ConvNetLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3645948244034058451L;

	private ConvolutionalLayer cl;

	private NonlinearityLayer nl;

	private MaxPoolingLayer pl;

	private int emb_size;

	private int num_filters;

	private int output_size;

	private int window_size;

	public ConvNetLayer(ConvolutionalLayer cl, NonlinearityLayer nl, MaxPoolingLayer pl) {
		this.cl = cl;
		this.nl = nl;
		this.pl = pl;

		emb_size = cl.getEmbeddingSize();
		num_filters = cl.getInputSize();
		window_size = cl.getWindowSize();
	}

	public ConvNetLayer(int emb_size, int window_size, int num_filters) {
		this.emb_size = emb_size;
		this.num_filters = num_filters;
		this.window_size = window_size;

		cl = new ConvolutionalLayer(emb_size, window_size, num_filters);

		nl = new NonlinearityLayer(new ReLU());

		pl = new MaxPoolingLayer(num_filters);
	}

	@Override
	public Object backward(Object I) {
		Object O = cl.backward(I);
		O = nl.backward(O);
		O = pl.backward(O);
		return O;
	}

	@Override
	public Layer copy() {
		return new ConvNetLayer(cl, nl, pl);
	}

	@Override
	public Object forward(Object I) {
		Object O = cl.forward(I);
		O = nl.forward(O);
		O = pl.forward(O);
		return O;
	}

	@Override
	public DenseTensor getB() {
		return cl.getB();
	}

	@Override
	public DenseTensor getDB() {
		return cl.getDB();
	}

	@Override
	public DenseTensor getDW() {
		return cl.getDW();
	}

	@Override
	public int getInputSize() {
		return emb_size;
	}

	@Override
	public int getOutputSize() {
		return num_filters;
	}

	@Override
	public DenseTensor getW() {
		return cl.getW();
	}

	@Override
	public void initWeights() {
		cl.initWeights();
	}

	@Override
	public void prepareTraining() {
		cl.prepareTraining();
	}

}
