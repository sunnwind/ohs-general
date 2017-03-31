package ohs.ml.neuralnet.com;

public class NeuralNetParams {

	private int input_size = 10;

	private int hidden_size = 5;

	private int output_size = 10;

	private double learn_rate = 0.01;

	private double reg_lambda = 0.1;

	private int batch_size = 50;

	private int thread_size = 10;

	private double momentum = 0.9;

	private int bppt_size = 5;

	private int annealing_size = 20000;

	private double g_clip_threshold = 2;

	private int bptt_size = 5;

	public NeuralNetParams() {

	}

	public NeuralNetParams(int input_size, int hidden_size, int output_size, double learning_rate, int mini_batch_size, int thread_size) {
		super();
		this.input_size = input_size;
		this.hidden_size = hidden_size;
		this.output_size = output_size;
		this.learn_rate = learning_rate;
		this.batch_size = mini_batch_size;
		this.thread_size = thread_size;
	}

	public int getAnnealingSize() {
		return annealing_size;
	}

	public int getBpttSize() {
		return bppt_size;
	}

	public double getGradientClippingThreshold() {
		return g_clip_threshold;
	}

	public int getHiddenSize() {
		return hidden_size;
	}

	public int getInputSize() {
		return input_size;
	}

	public double getLearnRate() {
		return learn_rate;
	}

	public int getBatchSize() {
		return batch_size;
	}

	public double getMomentum() {
		return momentum;
	}

	public int getOutputSize() {
		return output_size;
	}

	public double getRegLambda() {
		return reg_lambda;
	}

	public int getThreadSize() {
		return thread_size;
	}

	public void setAnnealingSize(int bppt_size) {
		this.bppt_size = bppt_size;
	}

	public void setBpttSize(int bppt_size) {
		this.bppt_size = bppt_size;
	}

	public void setGradientClippingThreshold(double g_clip_threshold) {
		this.g_clip_threshold = g_clip_threshold;
	}

	public void setHiddenSize(int hidden_size) {
		this.hidden_size = hidden_size;
	}

	public void setInputSize(int input_size) {
		this.input_size = input_size;
	}

	public void setLearnRate(double learning_rate) {
		this.learn_rate = learning_rate;
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public void setOutputSize(int output_size) {
		this.output_size = output_size;
	}

	public void setRegLambda(double regularize_mixture) {
		this.reg_lambda = regularize_mixture;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		return sb.toString();
	}

}
