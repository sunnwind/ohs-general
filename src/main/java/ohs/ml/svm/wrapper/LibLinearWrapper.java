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
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.eval.Performance;
import ohs.ml.eval.PerformanceEvaluator;
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

	private Indexer<String> labelIdxer;

	private Indexer<String> featIdxer;

	public LibLinearWrapper() {

	}

	public LibLinearWrapper(Model model, Indexer<String> labelIdxer, Indexer<String> featIdxer) {
		this.model = model;
		this.labelIdxer = labelIdxer;
		this.featIdxer = featIdxer;
	}

	public Performance evalute(SparseMatrix X, DenseVector Y) {
		DenseVector Yh = new DenseVector(Y.size());
		for (int i = 0; i < X.rowSize(); i++) {
			Yh.add(i, score(X.row(i)).argMax());
		}
		return new PerformanceEvaluator().evalute(Y, Yh, labelIdxer);
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}

	public Indexer<String> getLabelIndexer() {
		return labelIdxer;
	}

	public Model getModel() {
		return model;
	}

	public void read(String fileName) throws Exception {
		System.out.printf("read at [%s]\n", fileName);
		BufferedReader br = FileUtils.openBufferedReader(fileName);
		labelIdxer = FileUtils.readStringIndexerFromText(br);
		featIdxer = FileUtils.readStringIndexerFromText(br);
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
		SparseVector sv = VectorUtils.toSparseVector(x, featIdxer, false);
		VectorMath.unitVector(sv);
		return VectorUtils.toCounter(score(sv), labelIdxer);
	}

	public SparseMatrix score(SparseMatrix X) {
		List<SparseVector> ret = new ArrayList<SparseVector>(X.rowSize());
		for (int i = 0; i < X.size(); i++) {
			ret.add(score(X.get(i)));
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
		FileUtils.writeStringIndexerAsText(bw, labelIdxer);
		FileUtils.writeStringIndexerAsText(bw, featIdxer);
		model.save(bw);
		bw.close();
	}

}
