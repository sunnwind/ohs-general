package ohs.ml.word2vec;

public class Word2VecParam {

	public static enum CostType {
		SOFTMAX, NEG_SAMPLE_COST
	}

	public static enum NetworkType {
		SKIP_GRAM, CBOW
	}

	private int vocab_size = 10;

	private int hidden_size = 10;

	private int context_size = 5;

	private int neg_sample_size = 10;

	private double learning_rate = 0.3;

	private int num_threads = 5;

	private int mini_batch_size = 50;

	private int annealing_size = 10000;

	private NetworkType networkType = NetworkType.SKIP_GRAM;

	private CostType costType = CostType.NEG_SAMPLE_COST;

	public Word2VecParam(NetworkType networkType, CostType costType, int vocab_size, int hidden_size, int context_size) {
		this.networkType = networkType;
		this.costType = costType;
		this.vocab_size = vocab_size;
		this.hidden_size = hidden_size;
		this.context_size = context_size;
	}

	public int getAnnealingSize() {
		return annealing_size;
	}

	public int getContextSize() {
		return context_size;
	}

	public CostType getCostType() {
		return costType;
	}

	public int getHiddenSize() {
		return hidden_size;
	}

	public double getLearningRate() {
		return learning_rate;
	}

	public int getMiniBatchSize() {
		return mini_batch_size;
	}

	public int getNegSampleSize() {
		return neg_sample_size;
	}

	public int getNumThreads() {
		return num_threads;
	}

	public void setHiddenSize(int hidden_size) {
		this.hidden_size = hidden_size;
	}

	public NetworkType getType() {
		return networkType;
	}

	public int getVocabSize() {
		return vocab_size;
	}

	public void setAnnealingSize(int annealing_size) {
		this.annealing_size = annealing_size;
	}

	public void setContextSize(int context_size) {
		this.context_size = context_size;
	}

	public void setCostType(CostType costType) {
		this.costType = costType;
	}

	public void setLearningRate(double learning_rate) {
		this.learning_rate = learning_rate;
	}

	public void setMiniBatchSize(int mini_batch_size) {
		this.mini_batch_size = mini_batch_size;
	}

	public void setNegSampleSize(int neg_sample_size) {
		this.neg_sample_size = neg_sample_size;
	}

	public void setNetworkType(NetworkType networkType) {
		this.networkType = networkType;
	}

	public void setNumThreads(int num_threads) {
		this.num_threads = num_threads;
	}

}
