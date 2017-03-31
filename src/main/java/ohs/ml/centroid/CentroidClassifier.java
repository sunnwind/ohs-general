package ohs.ml.centroid;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;

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

	public static CentroidClassifier train(List<SparseVector> trainData, List<Integer> labels) {
		ListMap<Integer, SparseVector> labelDocs = new ListMap<Integer, SparseVector>();
		Counter<Integer> docFreqs = new Counter<Integer>();

		TermWeighting.tfidf(trainData);

		for (int i = 0; i < trainData.size(); i++) {
			labelDocs.put(labels.get(i), trainData.get(i));
		}

		Map<Integer, SparseVector> centroids = new TreeMap<Integer, SparseVector>();
		SparseVector labelPriors = new SparseVector(labelDocs.size());

		for (int i = 0; i < labels.size(); i++) {
			int label = labels.get(i);
			List<SparseVector> docs = labelDocs.get(label);
			Counter<Integer> c = new Counter<Integer>();
			for (SparseVector doc : docs) {
				for (int j = 0; j < doc.size(); j++) {
					int w = doc.indexAt(j);
					double weight = doc.valueAt(j);
					c.incrementCount(w, weight);
				}
			}
			SparseVector centroid = VectorUtils.toSparseVector(c);
			centroid.multiply(1f / docs.size());

			centroids.put(label, centroid);
			labelPriors.addAt(i, label, docs.size());
		}

		labelPriors.normalizeAfterSummation();
		VectorMath.unitVector(labelPriors);

		return new CentroidClassifier(null, null, centroids, labelPriors, 1, KernelType.LINEAR, 1f / docFreqs.size(), 0, 3);
	}

	private Map<Integer, SparseVector> centroids;

	private double gamma;

	private double coef0;

	private int degree;

	private int scoreType;

	private SparseVector labelBias;

	private KernelType kernelType;

	private Indexer featIndexer;

	private Indexer labelIndexer;

	public CentroidClassifier(Indexer<String> labelIndexer, Indexer<String> featIndexer,

			Map<Integer, SparseVector> centroids, SparseVector labelBias, int scoreType, KernelType kernelType, double gamma, double coef0,
			int degree) {
		this.labelBias = labelBias;

		this.centroids = centroids;

		this.scoreType = scoreType;

		this.kernelType = kernelType;

		this.gamma = gamma;

		this.coef0 = coef0;

		this.degree = degree;
	}

	public SparseVector getCentroid(int label) {
		return centroids.get(label);
	}

	public Indexer getFeatureIndexer() {
		return featIndexer;
	}

	public Indexer getLabelIndexer() {
		return labelIndexer;
	}

	public List<SparseVector> score(List<SparseVector> inputData) {
		List<SparseVector> outputData = new ArrayList<SparseVector>();
		for (int i = 0; i < inputData.size(); i++) {
			SparseVector input = inputData.get(i);
			SparseVector output = score(input);
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
	// Counter<String> topic_score = VectorUtils.toCounter(output, info.topicIndexer());
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
	// Counter<String> topic_score = VectorUtils.toCounter(score(term_count), info.topicIndexer());
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

	public SparseVector score(SparseVector x) {
		return score(x, centroids.keySet());
	}

	public SparseVector score(SparseVector x, Set<Integer> labelSet) {
		SparseVector ret = new SparseVector(labelSet.size());
		int i = 0;
		for (int label : labelSet) {
			SparseVector c = centroids.get(label);
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
			ret.addAt(i++, label, sim);
		}
		return ret;
	}

	public void setFeatureIndexer(Indexer featIndexer) {
		this.featIndexer = featIndexer;
	}

	public void setLabelIndexer(Indexer labelIndexer) {
		this.labelIndexer = labelIndexer;
	}

	public void setTopicBias(SparseVector topic_bias) {
		this.labelBias = topic_bias;
	}

}
