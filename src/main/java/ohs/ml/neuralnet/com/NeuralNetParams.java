package ohs.ml.neuralnet.com;

import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;

public class NeuralNetParams {

	private double grad_clip_cutoff = Double.MAX_VALUE;

	private double learn_rate = 0.01;

	private double learn_rate_decay = 0.9;

	private double momentum = 0.9;

	private double reg_lambda = 0.1;

	private double weight_decay_L2 = 1;

	private double grad_decay = 1d / 1000000;

	private int annealing_size = 20000;

	private int batch_size = 50;

	private int k1 = 3;

	private int k2 = 3;

	private int grad_acc_reset_size = 100;

	private int hidden_size = 5;

	private int input_size = 10;

	private int learn_rate_decay_size = 1000;

	private int output_size = 10;

	private int thread_size = 10;

	private boolean is_full_seq_batch = false;

	private boolean is_random_batch = true;

	private boolean use_avg_grad = false;

	private boolean use_hard_grad_clipping = false;

	private OptimizerType ot = OptimizerType.ADAM;

	public NeuralNetParams() {

	}

	public NeuralNetParams(int input_size, int hidden_size, int output_size, double learning_rate, int mini_batch_size,
			int thread_size) {
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

	public int getBatchSize() {
		return batch_size;
	}

	public int getGradientAccumulatorResetSize() {
		return grad_acc_reset_size;
	}

	public double getGradientClipCutoff() {
		return grad_clip_cutoff;
	}

	public double getGradientDecay() {
		return grad_decay;
	}

	public int getHiddenSize() {
		return hidden_size;
	}

	public int getInputSize() {
		return input_size;
	}

	public int getK1() {
		return k1;
	}

	public int getK2() {
		return k2;
	}

	public double getLearnRate() {
		return learn_rate;
	}

	public double getLearnRateDecay() {
		return learn_rate_decay;
	}

	public int getLearnRateDecaySize() {
		return learn_rate_decay_size;
	}

	public double getMomentum() {
		return momentum;
	}

	public OptimizerType getOptimizerType() {
		return ot;
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

	public boolean getUseAverageGradients() {
		return use_avg_grad;
	}

	public boolean getUseHardGradientClipping() {
		return use_hard_grad_clipping;
	}

	public double getWeightDecayL2() {
		return weight_decay_L2;
	}

	public boolean isFullSequenceBatch() {
		return is_full_seq_batch;
	}

	public boolean isRandomBatch() {
		return is_random_batch;
	}

	public void setBatchSize(int batch_size) {
		this.batch_size = batch_size;
	}

	public void setGradientAccumulatorResetSize(int grad_acc_reset_size) {
		this.grad_acc_reset_size = grad_acc_reset_size;
	}

	public void setGradientClipCutoff(double grad_clip_cutoff) {
		this.grad_clip_cutoff = grad_clip_cutoff;
	}

	public void setGradientDecay(double grad_decay) {
		this.grad_decay = grad_decay;
	}

	public void setHiddenSize(int hidden_size) {
		this.hidden_size = hidden_size;
	}

	public void setInputSize(int input_size) {
		this.input_size = input_size;
	}

	public void setIsFullSequenceBatch(boolean is_full_seq_batch) {
		this.is_full_seq_batch = is_full_seq_batch;
	}

	public void setIsRandomBatch(boolean is_random_batch) {
		this.is_random_batch = is_random_batch;
	}

	public void setK1(int k1) {
		this.k1 = k1;
	}

	public void setK2(int k2) {
		this.k2 = k2;
	}

	public void setLearnRate(double learning_rate) {
		this.learn_rate = learning_rate;
	}

	public void setLearnRateDecay(double learn_rate_decay) {
		this.learn_rate_decay = learn_rate_decay;
	}

	public void setLearnRateDecaySize(int learn_rate_decay_size) {
		this.learn_rate_decay_size = learn_rate_decay_size;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public void setOptimizerType(OptimizerType ot) {
		this.ot = ot;
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

	public void setUseAverageGradients(boolean use_avg_grad) {
		this.use_avg_grad = use_avg_grad;
	}

	public void setUseHardGradientClipping(boolean use_hard_grad_clipping) {
		this.use_hard_grad_clipping = use_hard_grad_clipping;
	}

	public void setWeightDecayL2(double weight_decay_L2) {
		this.weight_decay_L2 = weight_decay_L2;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		return sb.toString();
	}

}
