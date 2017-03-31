package ohs.ml.glove;

public class GloveParam {

	private int vocab_size = 10;

	private int hidden_size = 10;

	private double learn_rate = 0.3;

	private int thread_size = 5;

	private int mini_batch_size = 50;

	private int anneal_size = 10000;

	private double max_x = 100;

	private double alpha = 0.75;

	public GloveParam(int vocab_size, int hidden_size) {
		this.vocab_size = vocab_size;
		this.hidden_size = hidden_size;
	}

	public double getAlpha() {
		return alpha;
	}

	public int getAnnealingSize() {
		return anneal_size;
	}

	public int getHiddenSize() {
		return hidden_size;
	}

	public double getLearningRate() {
		return learn_rate;
	}

	public double getMaxX() {
		return max_x;
	}

	public int getMiniBatchSize() {
		return mini_batch_size;
	}

	public int getThreadSize() {
		return thread_size;
	}

	public int getVocabSize() {
		return vocab_size;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public void setAnnealingSize(int anneal_size) {
		this.anneal_size = anneal_size;
	}

	public void setLearnRate(double learn_rate) {
		this.learn_rate = learn_rate;
	}

	public void setMaxX(double max_x) {
		this.max_x = max_x;
	}

	public void setMiniBatchSize(int mini_batch_size) {
		this.mini_batch_size = mini_batch_size;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
