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

	public static CentroidClassifier train(List<SparseVector> X, List<Integer> Y) {
		ListMap<Integer, SparseVector> labelDocs = new ListMap<Integer, SparseVector>();
		Counter<Integer> docFreqs = new Counter<Integer>();

		TermWeighting.tfidf(X);

		for (int i = 0; i < X.size(); i++) {
			labelDocs.put(Y.get(i), X.get(i));
		}

		Map<Integer, SparseVector> C = new TreeMap<Integer, SparseVector>();
		SparseVector priors = new SparseVector(labelDocs.size());

		for (int i = 0; i < Y.size(); i++) {
			int label = Y.get(i);
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

			C.put(label, centroid);
			priors.addAt(i, label, docs.size());
		}

		priors.normalizeAfterSummation();
		VectorMath.unitVector(priors);

		return new CentroidClassifier(null, null, C, priors, 1, KernelType.LINEAR, 1f / docFreqs.size(), 0, 3);
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

	private Map<Integer, SparseVector> C;

	private double coef0;

	private int degree;

	private Indexer<String> featIdxer;

	private double gamma;

	private KernelType kernelType;

	private SparseVector labelBias;

	private Indexer<String> labelIdxer;

	private int scoreType;

	public CentroidClassifier(Indexer<String> labelIndexer, Indexer<String> featIndexer,

			Map<Integer, SparseVector> centroids, SparseVector labelBias, int scoreType, KernelType kernelType,
			double gamma, double coef0, int degree) {
		this.labelBias = labelBias;

		this.C = centroids;

		this.scoreType = scoreType;

		this.kernelType = kernelType;

		this.gamma = gamma;

		this.coef0 = coef0;

		this.degree = degree;
	}

	public SparseVector getCentroid(int label) {
		return C.get(label);
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}

	public Indexer<String> getLabelIndexer() {
		return labelIdxer;
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

	public SparseVector score(SparseVector x) {
		return score(x, C.keySet());
	}

	public SparseVector score(SparseVector x, Set<Integer> labelSet) {
		SparseVector ret = new SparseVector(labelSet.size());
		int i = 0;
		for (int label : labelSet) {
			SparseVector c = C.get(label);
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

	public void setFeatureIndexer(Indexer<String> featIndexer) {
		this.featIdxer = featIndexer;
	}

	public void setLabelIndexer(Indexer<String> labelIndexer) {
		this.labelIdxer = labelIndexer;
	}

	public void setTopicBias(SparseVector topic_bias) {
		this.labelBias = topic_bias;
	}

}
