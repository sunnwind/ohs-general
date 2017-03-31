package ohs.eden.keyphrase.mine;

import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.corpus.search.app.RandomAccessDenseMatrix;
import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Generics.ListType;

public class PhraseQualityClassification {

	public static String dir = MIRPath.TREC_CDS_2016_DIR;

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		PhraseQualityClassification qc = new PhraseQualityClassification();
		// qc.generateData();
		// qc.train();

		// qc.buildLabeledData();
		// qc.buildUnlabeledData();
		// qc.trainClassifier();
		qc.test();

		System.out.println("process ends.");
	}

	public String delim = "  ";

	private DenseVector extractFeatures(RandomAccessDenseMatrix E, Vocab vocab, String phrs) throws Exception {
		DenseVector ret = null;

		IntegerArray Q = vocab.indexesOf(phrs.split(" "));

		int unseen_cnt = 0;

		for (int w : Q) {
			if (w < 0) {
				unseen_cnt++;
			}
		}

		if (unseen_cnt > 0) {
			return ret;
		}

		DenseMatrix E2 = E.rowsAsMatrix(Q.values()).copy();

		Map<Integer, DenseVector> set = Generics.newHashMap();

		for (int i = 0; i < Q.size(); i++) {
			set.put(Q.get(i), E2.get(i));
		}

		DenseVector e_avg = new DenseVector(E.colSize());

		for (DenseVector e : E2) {
			VectorMath.add(e, e_avg);
		}

		e_avg.multiply(1f / Q.size());

		// VectorMath.unitVector(e_avg);

		List<Double> vals = Generics.newLinkedList();

		for (double val : E2.get(0).values()) {
			vals.add(val);
		}

		for (double val : E2.get(E2.rowSize() - 1).values()) {
			vals.add(val);
		}

		for (double val : e_avg.values()) {
			vals.add(val);
		}

		ret = new DenseVector(vals);
		return ret;
	}

	public void buildLabeledData() throws Exception {
		ListMap<String, String> lm = Generics.newListMap(ListType.LINKED_LIST);

		{
			Counter<String> c = FileUtils.readStringCounterFromText(dir + "phrs/phrs_good.txt");
			for (String phrs : c.getSortedKeys()) {
				lm.put("good", phrs);
			}
		}

		{
			Counter<String> c = FileUtils.readStringCounterFromText(dir + "phrs/phrs_bad.txt");
			c.keepTopNKeys(1000000);

			for (String phrs : c.getSortedKeys()) {
				lm.put("bad", phrs);
			}
		}

		DocumentCollection dc = new DocumentCollection(dir + "col/dc/");
		Vocab vocab = dc.getVocab();

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(dir + "emb/glove_ra.ser");
		// RandomAccessDenseMatrix E = new RandomAccessDenseMatrix("../../data/medical_ir/wiki/" + "emb/glove_ra.ser");

		List<DenseVector> data = Generics.newLinkedList();
		List<Integer> labels = Generics.newLinkedList();
		List<String> phrsData = Generics.newLinkedList();

		Counter<String> c = Generics.newCounter();

		for (String label : lm.keySet()) {
			List<String> phrss = lm.get(label);

			int l = 2;

			if (label.equals("good")) {
				l = 1;
			} else if (label.equals("bad")) {
				l = 0;
			}

			for (String phrs : phrss) {
				DenseVector e = extractFeatures(E, vocab, phrs);
				if (e == null) {
					continue;
				}
				data.add(e);
				labels.add(l);
				phrsData.add(label + "\t" + phrs);
			}
		}

		System.out.println(c.toString());

		new DenseMatrix(data).writeObject(dir + "phrs/data_labeled.ser.gz");
		new IntegerArray(labels).writeObject(dir + "phrs/label.ser.gz");
		FileUtils.writeStringCollection(dir + "phrs/phrs_labeled.txt", phrsData);
	}

	public void buildUnlabeledData() throws Exception {
		Counter<String> c = FileUtils.readStringCounterFromText(dir + "phrs/phrs_not_bad.txt");
		c.keepTopNKeys(500000);

		DocumentCollection dc = new DocumentCollection(dir + "col/dc/");
		Vocab vocab = dc.getVocab();

		DenseVector idfs = new DenseVector(vocab.size());
		for (int i = 0; i < vocab.size(); i++) {
			double idf = TermWeighting.idf(vocab.getDocCnt(), vocab.getDocFreq(i));
			idfs.add(i, idf);
		}

		// idfs.normalize();

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(dir + "emb/glove_ra.ser");
		// RandomAccessDenseMatrix E = new RandomAccessDenseMatrix("../../data/medical_ir/wiki/" + "emb/glove_ra.ser");

		List<DenseVector> data = Generics.newLinkedList();
		List<String> phrsData = Generics.newLinkedList();

		List<String> phrss = c.getSortedKeys();

		for (String phrs : phrss) {
			DenseVector e = extractFeatures(E, vocab, phrs);
			if (e == null) {
				continue;
			}
			data.add(e);
			phrsData.add(phrs);
		}

		new DenseMatrix(data).writeObject(dir + "phrs/data_unlabeled.ser.gz");
		FileUtils.writeStringCollection(dir + "phrs/phrs_unlabeled.txt", phrsData);
	}

	public void generateData() throws Exception {
		List<String> lines = FileUtils.readLinesFromText(KPPath.KYP_DIR + "phrs_3p_label.txt.gz");

		IntegerArrayMatrix I = new IntegerArrayMatrix();
		IntegerArray L = new IntegerArray();

		Vocab vocab = new Vocab();
		vocab.readObject(KPPath.COL_DC_DIR + "vocab.ser.gz");

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			List<String> parts = Generics.newArrayList(line.split("\t"));
			String phrs = parts.get(0);

			int phrs_cnt = Integer.parseInt(parts.get(1));
			int kwd_cnt = Integer.parseInt(parts.get(2));
			String quality = "X";

			if (kwd_cnt >= 10) {
				quality = "O";
			}

			parts.add(quality);

			lines.set(i, StrUtils.join("\t", parts));

			int label = quality.equals("X") ? 0 : 1;
			I.add(label, i);
			L.add(label);
		}

		I.trimToSize();

		for (IntegerArray a : I) {
			ArrayUtils.shuffle(a.values());
		}

		int train_size = 1000;
		int test_size = 20000;

		IntegerArrayMatrix D = DataSplitter.splitGroups(I, new int[] { train_size, test_size });

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int i = 0; i < D.size(); i++) {
			IntegerArray a = D.get(i);
			for (int j = 0; j < a.size(); j++) {
				int loc = a.get(j);
				int label = L.get(loc);
				cm.incrementCount(i == 0 ? "train" : "test", label == 0 ? "X" : "O", 1);
			}
		}

		System.out.println(cm);

		DenseMatrix X = new DenseMatrix();
		DenseMatrix Xt = new DenseMatrix();

		IntegerArray Y = new IntegerArray();
		IntegerArray Yt = new IntegerArray();

		List<String> S = Generics.newArrayList();
		List<String> St = Generics.newArrayList();

		DenseMatrix EW = new DenseMatrix();
		EW.readObject(KPPath.KYP_DIR + "glove_embedding.ser.gz");

		DenseMatrix ED = new DenseMatrix();
		ED.readObject(KPPath.KYP_DIR + "glove_embedding_doc.ser.gz");

		getData(lines, vocab, EW, ED, D.get(0), L, X, Y, S);
		getData(lines, vocab, EW, ED, D.get(1), L, Xt, Yt, St);

		CounterMap<String, String> c = Generics.newCounterMap();

		for (int ans : Yt) {
			c.incrementCount("test", ans == 0 ? "X" : "O", 1);
		}

		for (int ans : Y) {
			c.incrementCount("train", ans == 0 ? "X" : "O", 1);
		}

		System.out.println(c.toString());

		X.writeObject(KPPath.KYP_DIR + "phrs_quality_train_X.ser.gz");
		Y.writeObject(KPPath.KYP_DIR + "phrs_quality_train_Y.ser.gz");
		FileUtils.writeStringCollection(KPPath.KYP_DIR + "phrs_train_S.txt.gz", S);

		Xt.writeObject(KPPath.KYP_DIR + "phrs_quality_test_X.ser.gz");
		Yt.writeObject(KPPath.KYP_DIR + "phrs_quality_test_Y.ser.gz");
		FileUtils.writeStringCollection(KPPath.KYP_DIR + "phrs_test_S.txt.gz", St);
	}

	public void generateDocumentEmbedding() throws Exception {
		DenseMatrix E = new DenseMatrix();
		E.readObject(KPPath.KYP_DIR + "glove_embedding.ser.gz");

		DocumentCollection ldc = new DocumentCollection(KPPath.COL_DC_DIR);

		Vocab vocab = ldc.getVocab();
		int doc_size = ldc.size();
		int hidden_size = E.colSize();

		DenseMatrix DE = new DenseMatrix(doc_size, hidden_size);

		for (int i = 0, j = 0; i < ldc.size(); i++) {
			IntegerArrayMatrix doc = ldc.getSents(i).getSecond();
			DenseVector de = DE.row(j++);
			int word_cnt = 0;

			for (IntegerArray sent : doc) {
				for (int w : sent) {
					DenseVector we = E.row(w);
					VectorMath.add(we, de);
					word_cnt++;
				}
			}

			de.multiply(1f / word_cnt);
		}

		DE.writeObject(KPPath.KYP_DIR + "glove_embedding_doc.ser.gz");
	}

	public void getData(List<String> lines, Vocab vocab, DenseMatrix EW, DenseMatrix ED, IntegerArray dlocs, IntegerArray L, DenseMatrix X,
			IntegerArray Y, List<String> S) {
		int hidden_size = EW.colSize();

		DenseVector f1 = new DenseVector(hidden_size);
		DenseVector f2 = new DenseVector(hidden_size);
		DenseVector f3 = new DenseVector(hidden_size);

		for (int i = 0; i < dlocs.size(); i++) {
			int dloc = dlocs.get(i);
			String line = lines.get(dloc);
			String[] parts = line.split("\t");
			String phrs = parts[0];
			String[] toks = phrs.split(delim);

			f1.setAll(0);
			f2.setAll(0);
			f3.setAll(0);

			boolean skip = false;

			for (int j = 0; j < toks.length; j++) {
				String tok = toks[j];
				int idx = vocab.indexOf(tok);

				if (idx < 0) {
					skip = true;
					break;
				}

				DenseVector e = EW.row(idx);

				if (j == 0) {
					VectorMath.add(e, f1);
				}

				if (j == toks.length - 1) {
					VectorMath.add(e, f3);
				}

				VectorMath.add(e, f2);
			}

			if (skip) {
				continue;
			}

			f2.multiply(1f / toks.length);

			DenseVector ed = ED.row(dloc);

			DenseVector x = new DenseVector(hidden_size * 2);

			for (int j = 0; j < f1.size(); j++) {
				// x.add(j, f1.value(j));
				x.add(j, f2.value(j));
				// x.add(hidden_size + j, f3.value(j));
				x.add(hidden_size + j, ed.value(j));
			}

			// DenseVector x = new DenseVector(hidden_size * 4);
			//
			// for (int j = 0; j < f1.size(); j++) {
			// x.add(j, f1.value(j));
			// x.add(hidden_size + j, f2.value(j));
			// x.add(hidden_size * 2 + j, f3.value(j));
			// x.add(hidden_size * 3 + j, ed.value(j));
			// }

			X.add(x);

			int y = L.get(dloc);

			Y.add(y);

			S.add(line);
		}

		X.trimToSize();
		Y.trimToSize();
	}

	public void train() throws Exception {
		NeuralNetParams param = new NeuralNetParams();
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(2);

		DenseMatrix X = new DenseMatrix();
		X.readObject(KPPath.KYP_DIR + "phrs_quality_train_X.ser.gz");

		IntegerArray Y = new IntegerArray();
		Y.readObject(KPPath.KYP_DIR + "phrs_quality_train_Y.ser.gz");

		DenseMatrix Xt = new DenseMatrix();
		Xt.readObject(KPPath.KYP_DIR + "phrs_quality_test_X.ser.gz");

		IntegerArray Yt = new IntegerArray();
		Yt.readObject(KPPath.KYP_DIR + "phrs_quality_test_Y.ser.gz");

		Counter<Integer> labels = Generics.newCounter();
		for (int y : Y) {
			labels.incrementCount(y, 1);
		}

		Counter<Integer> labels2 = Generics.newCounter();
		for (int y : Yt) {
			labels2.incrementCount(y, 1);
		}

		int input_size = X.colSize();
		int l1_size = 200;
		int l2_size = 50;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(input_size, l1_size));
		nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(l1_size, new Tanh()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(l2_size, new Tanh()));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));

		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, X.rowSize(), null);
		trainer.train(X, Y, Xt, Yt, 100);
		trainer.finish();

		List<String> lines = FileUtils.readLinesFromText(KPPath.KYP_DIR + "phrs_test_S.txt.gz");

		IntegerArray Yth = nn.classify(Xt);

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			List<String> parts = Generics.newArrayList(line.split("\t"));

			int p = Yth.get(i);
			int a = Yt.get(i);

			String pred = p == 0 ? "X" : "O";
			String ans = a == 0 ? "X" : "O";
			parts.add(pred);

			if (p == a) {
				parts.add("Yes");
			} else {
				parts.add("No");
			}
			// parts.add(ans);

			line = StrUtils.join("\t", parts);
			lines.set(i, line);
		}

		FileUtils.writeStringCollection(KPPath.KYP_DIR + "phrs_test_S_res.txt.gz", lines);
	}

	public void trainClassifier() throws Exception {

		DenseMatrix XD = new DenseMatrix(dir + "phrs/data_labeled.ser.gz");
		IntegerArray YD = new IntegerArray(dir + "phrs/label.ser.gz");

		DenseMatrix X = new DenseMatrix();
		IntegerArray Y = new IntegerArray();

		DenseMatrix Xt = new DenseMatrix();
		IntegerArray Yt = new IntegerArray();

		int test_size = 1000;

		{
			IntegerArrayMatrix G = DataSplitter.groupByLabels(YD);

			for (int i = 0; i < G.size(); i++) {
				IntegerArray locs = G.get(i);
				ArrayUtils.shuffle(locs.values());

				IntegerArray sub1 = locs.subArray(0, test_size);
				IntegerArray sub2 = locs.subArray(test_size, locs.size());

				for (int loc : sub1) {
					Xt.add(XD.get(loc));
					Yt.add(YD.get(loc));
				}

				for (int loc : sub2) {
					X.add(XD.get(loc));
					Y.add(YD.get(loc));
				}
			}
			X.trimToSize();
			Y.trimToSize();
			Xt.trimToSize();
			Yt.trimToSize();

			X.unwrapValues();
			Xt.unwrapValues();
		}

		Set<Integer> labels = Generics.newHashSet();
		for (int y : Y) {
			labels.add(y);
		}

		NeuralNetParams param = new NeuralNetParams();
		param.setInputSize(100);
		param.setHiddenSize(50);
		param.setOutputSize(10);
		param.setBatchSize(10);
		param.setLearnRate(0.001);
		param.setRegLambda(0.001);
		param.setThreadSize(4);

		int feat_size = X.colSize();
		int l1_size = 200;
		int l2_size = 50;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(feat_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(l1_size, new Tanh()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(l2_size, new Tanh()));
		// nn.add(new DropoutLayer(l2_size));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, XD.rowSize(), null);

		IntegerArrayMatrix G = DataSplitter.groupByLabels(Y);
		IntegerArray negLocs = G.get(0);
		IntegerArray posLocs = G.get(1);

		for (int i = 0; i < 1; i++) {
			ArrayUtils.shuffle(negLocs.values());
			int pos_size = posLocs.size();

			for (int j = 0; j < negLocs.size();) {
				int k = Math.min(negLocs.size(), j + pos_size);
				IntegerArray locs = new IntegerArray(pos_size * 2);

				for (int l = j; l < k; l++) {
					locs.add(negLocs.get(l));
				}
				j = k;

				for (int loc : posLocs) {
					locs.add(loc);
				}

				locs.trimToSize();

				DenseMatrix Xs = X.rowsAsMatrix(locs.values());
				IntegerArray Ys = Y.subArray(locs.values());
				trainer.train(Xs, Ys, Xt, Yt, 1);
			}
		}

		trainer.finish();

		nn.writeObject(dir + "phrs/quality_model.ser.gz");
	}

	public void test() throws Exception {
		NeuralNet nn = new NeuralNet();
		nn.readObject(dir + "/phrs/quality_model.ser.gz");

		DenseMatrix X = new DenseMatrix(dir + "phrs/data_unlabeled.ser.gz");
		List<String> phrss = FileUtils.readLinesFromText(dir + "phrs/phrs_unlabeled.txt");
		Counter<String> c = Generics.newCounter();
		
		for (int i = 0; i < X.rowSize(); i++) {
			DenseVector x = X.row(i);
			DenseVector scores = nn.score(x);
			int pred = scores.argMax();

			String label = pred == 0 ? "bad" : "good";
			String phrs = phrss.get(i);
			phrss.set(i, label + "\t" + phrs);
			c.incrementCount(label, 1);
		}
		
		System.out.println(c.toString());

		FileUtils.writeStringCollection(dir + "phrs/phrs_unlabeled_labeled.txt", phrss);

	}

}
