package ohs.ml.svm.wrapper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;

public class LibLinearWrapper implements Serializable {

	private static final long serialVersionUID = -3273222430839071709L;

	public static Feature[] toFeatures(DenseVector x, int num_feats, double bias) {
		Feature[] fs = new Feature[bias >= 0 ? x.size() + 1 : x.size()];
		for (int j = 0; j < x.size(); j++) {
			int index = x.indexAt(j);
			double value = x.valueAt(j);
			fs[j] = new FeatureNode(index + 1, value);
		}

		if (bias >= 0) {
			fs[fs.length - 1] = new FeatureNode(num_feats, bias);
		}
		return fs;
	}

	public static Feature[] toFeatures(SparseVector x, int num_feats, double bias) {
		Feature[] fs = new Feature[bias >= 0 ? x.size() + 1 : x.size()];
		for (int j = 0; j < x.size(); j++) {
			int index = x.indexAt(j);
			double value = x.valueAt(j);
			fs[j] = new FeatureNode(index + 1, value);
		}

		if (bias >= 0) {
			fs[fs.length - 1] = new FeatureNode(num_feats, bias);
		}
		return fs;
	}

	private Model model;

	private Indexer<String> labelIndexer;

	private Indexer<String> featIndexer;

	public LibLinearWrapper() {

	}

	public LibLinearWrapper(Model model, Indexer<String> labelIndexer, Indexer<String> featIndexer) {
		this.model = model;
		this.labelIndexer = labelIndexer;
		this.featIndexer = featIndexer;
	}

	public String evalute(SparseMatrix X, IntegerArray Y) {
		SparseVector correct = new SparseVector(ArrayUtils.copy(model.getLabels()));
		correct.sortIndexes();

		SparseVector anss = correct.copy();
		SparseVector preds = correct.copy();

		for (int i = 0; i < X.size(); i++) {
			SparseVector x = X.get(i);
			SparseVector scores = score(x);
			int pred = scores.argMax();
			int ans = Y.get(i);

			if (pred == ans) {
				correct.add(ans, 1);
			}

			anss.add(ans, 1);
			preds.add(pred, 1);
		}

		return TopicEval.evalute(null, anss, preds, correct);
	}

	public Indexer<String> getFeatureIndexer() {
		return featIndexer;
	}

	public Indexer<String> getLabelIndexer() {
		return labelIndexer;
	}

	public Model getModel() {
		return model;
	}

	public void read(String fileName) throws Exception {
		System.out.printf("read at [%s]\n", fileName);
		BufferedReader br = FileUtils.openBufferedReader(fileName);
		labelIndexer = FileUtils.readStringIndexerFromText(br);
		featIndexer = FileUtils.readStringIndexerFromText(br);
		model = Linear.loadModel(br);
		br.close();

	}

	public double regression(DenseVector x) {
		int[] labels = new int[model.getNrClass()];
		double[] ret = new double[labels.length];
		Feature[] fs = toFeatures(x, model.getNrFeature(), model.getBias());
		Linear.predictValues(model, fs, ret);
		return ret[0];
	}

	public Counter<String> score(Counter<String> x) {
		SparseVector sv = VectorUtils.toSparseVector(x, featIndexer, false);
		VectorMath.unitVector(sv);

		return VectorUtils.toCounter(score(sv), labelIndexer);
	}

	public SparseMatrix score(SparseMatrix X) {
		List<SparseVector> ret = new ArrayList<SparseVector>();
		for (int i = 0; i < X.size(); i++) {
			SparseVector x = X.get(i);
			SparseVector scores = score(x);
			ret.add(scores);
		}
		return new SparseMatrix(ret);
	}

	public SparseVector score(SparseVector x) {
		int[] labels = model.getLabels();
		double[] scores = new double[labels.length];
		Feature[] fs = toFeatures(x, model.getNrFeature(), model.getBias());

		Linear.predictValues(model, fs, scores);

		SparseVector ret = new SparseVector(labels, scores);
		VectorMath.softmax(ret);
		return ret;
	}

	public void write(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		BufferedWriter bw = FileUtils.openBufferedWriter(fileName);
		FileUtils.writeStringIndexerAsText(bw, labelIndexer);
		FileUtils.writeStringIndexerAsText(bw, featIndexer);
		model.save(bw);
		bw.close();
	}

}
