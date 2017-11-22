package ohs.ml.centroid;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;

public class CentroidClassifier {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		System.out.println("process ends.");
	}

	private static double powi(double base, int times) {
		double tmp = base, ret = 1.0;
		for (int t = times; t > 0; t /= 2) {
			if (t % 2 == 1)
				ret *= tmp;
			tmp = tmp * tmp;
		}
		return ret;
	}

	public static CentroidClassifier train(SparseMatrix X, DenseVector Y, Indexer<String> labelIdxer) {
		ListMap<Integer, SparseVector> labelDocs = new ListMap<Integer, SparseVector>();
		Counter<Integer> docFreqs = new Counter<Integer>();

		TermWeighting.tfidf(X);

		for (int i = 0; i < X.size(); i++) {
			labelDocs.put((int) Y.value(i), X.get(i));
		}

		Map<Integer, SparseVector> C = new TreeMap<Integer, SparseVector>();
		DenseVector priors = new DenseVector(labelDocs.size());

		for (int label : labelDocs.keySet()) {
			List<SparseVector> docs = labelDocs.get(label);
			Counter<Integer> tmp = new Counter<Integer>();

			for (SparseVector doc : docs) {
				VectorMath.add(doc, tmp);
			}

			SparseVector c = new SparseVector(tmp);
			c.multiply(1d / docs.size());

			C.put(label, c);
			priors.add(label, docs.size());
		}

		priors.normalizeAfterSummation();

		return new CentroidClassifier(null, null, new SparseMatrix(C), priors, 1, KernelType.LINEAR,
				1f / docFreqs.size(), 0, 3);
	}

	private Indexer<String> labelIdxer;

	private Indexer<String> featIdxer;

	private SparseMatrix C;

	private DenseVector priors;

	private double coef0;

	private int degree;

	private double gamma;

	private KernelType kernelType;

	private int scoreType;

	public CentroidClassifier(Indexer<String> labelIdxer, Indexer<String> featIdxer, SparseMatrix C, DenseVector priors,
			int scoreType, KernelType kernelType, double gamma, double coef0, int degree) {

		this.priors = priors;

		this.C = C;

		this.scoreType = scoreType;

		this.kernelType = kernelType;

		this.gamma = gamma;

		this.coef0 = coef0;

		this.degree = degree;
	}

	public SparseVector getCentroid(int label) {
		return C.row(label);
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}

	public Indexer<String> getLabelIndexer() {
		return labelIdxer;
	}

	public List<DenseVector> score(List<SparseVector> inputData) {
		List<DenseVector> outputData = Generics.newLinkedList();
		for (int i = 0; i < inputData.size(); i++) {
			SparseVector input = inputData.get(i);
			DenseVector output = score(input);
			outputData.add(output);
		}
		return outputData;
	}

	// public TopicEval evaluate(List<SparseVector> testData) {
	// return evaluation(score(testData));
	// }

	// public TopicEval evaluation(List<SparseVector> outputData) {
	// List<String> answers = new ArrayList<String>();
	// List<Counter<String>> allPredicts = new ArrayList<Counter<String>>();
	//
	// for (int i = 0; i < outputData.size(); i++) {
	// SparseVector output = outputData.get(i);
	// Counter<String> topic_score = VectorUtils.toCounter(output,
	// info.topicIndexer());
	// String answer = info.topicIndexer().getObject(output.label());
	//
	// answers.add(answer);
	// allPredicts.add(topic_score);
	// }
	//
	// TopicEval topicEval = new TopicEval(answers, allPredicts);
	// topicEval.format();
	// return topicEval;
	// }

	// public TopicEval evalute(List<SparseVector> testData) {
	// List<String> answers = new ArrayList<String>();
	// List<Counter<String>> allPredicts = new ArrayList<Counter<String>>();
	//
	// for (int i = 0; i < testData.size(); i++) {
	// SparseVector term_count = testData.get(i);
	// Counter<String> topic_score = VectorUtils.toCounter(score(term_count),
	// info.topicIndexer());
	// String answer = info.topicIndexer().getObject(term_count.label());
	//
	// answers.add(answer);
	// allPredicts.add(topic_score);
	// }
	//
	// TopicEval topicEval = new TopicEval(answers, allPredicts);
	// topicEval.format();
	// return topicEval;
	// }

	public DenseVector score(SparseVector x) {
		DenseVector ret = new DenseVector(C.rowSize());
		for (int i = 0; i < C.rowSize(); i++) {
			SparseVector c = C.rowAt(i);
			double sim = 0;
			double dot_product = 0;

			switch (kernelType) {
			case LINEAR:
				sim = VectorMath.dotProduct(x, c);
				break;
			case POLY:
				dot_product = VectorMath.dotProduct(x, c);
				sim = powi(gamma * dot_product + coef0, degree);
				break;
			case SIGMOD:
				dot_product = VectorMath.dotProduct(x, c);
				sim = Math.tanh(gamma * dot_product + coef0);
			default:
				break;
			}
			ret.add(i, sim);
		}
		return ret;
	}

	public void setFeatureIndexer(Indexer<String> featIndexer) {
		this.featIdxer = featIndexer;
	}

	public void setLabelIndexer(Indexer<String> labelIndexer) {
		this.labelIdxer = labelIndexer;
	}

	public void setLabelPriors(DenseVector priors) {
		this.priors = priors;
	}

}
