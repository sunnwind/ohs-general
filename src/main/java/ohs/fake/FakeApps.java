package ohs.fake;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.eval.Performance;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.com.TaskType;
import ohs.ml.neuralnet.layer.ConcatenationLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.MultiWindowConvolutionalLayer;
import ohs.ml.neuralnet.layer.MultiWindowMaxPoolingLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class FakeApps {

	public static NewsFeatureExtractor getFeatureExtractor(String extFileName, LDocumentCollection X) throws Exception {
		NewsFeatureExtractor ext = new NewsFeatureExtractor();

		if (extFileName != null) {
			ext.readObject(extFileName);
		} else {
			Set<String> pers = Generics.newHashSet();
			Set<String> orgs = Generics.newHashSet();

			{
				String dirName = "../../data/fake_news/dict/";
				pers = FileUtils.readStringHashSetFromText(dirName + "pers.txt");
				orgs = FileUtils.readStringHashSetFromText(dirName + "orgs.txt");
			}

			Set<String> pol = Generics.newHashSet();
			Set<String> eco = Generics.newHashSet();

			{
				String dirName = "../../data/fake_news/dict_topic/";
				pol = FileUtils.readStringCounterFromText(dirName + "정치.txt").keySet();
				eco = FileUtils.readStringCounterFromText(dirName + "경제.txt").keySet();
			}

			Set<String> poss = Generics.newHashSet(X.getTokenStrings(1));
			Set<String> prefixes = Generics.newHashSet();

			LibLinearWrapper llw = new LibLinearWrapper();
			llw.read("../../data/fake_news/topic_model.txt");

			{
				Counter<String> c = Generics.newCounter();
				for (String word : X.getTokenStrings(0)) {
					if (word.length() > 3) {
						String p = word.substring(0, 3);
						p = p.toLowerCase();

						Set<Integer> l = Generics.newHashSet();

						for (int i = 0; i < p.length(); i++) {
							if (!Character.isDigit(p.charAt(i))) {
								l.add(i);
							}
						}

						if (l.size() == p.length()) {
							c.incrementCount(p, 1);
						}
					}
				}
				c.pruneKeysBelowThreshold(5);
				prefixes = c.keySet();
			}

			ext.addPosFeatures(poss);
			// ext.addCapitalFeatures();
			// ext.addPuctuationFeatures();
			// ext.addShapeOneFeatures();
			// ext.addShapeTwoFeatures();
			// ext.addShapeThreeFeatures();
			// ext.addSuffixFeatures(suffixes);
			// ext.addPrefixFeatures(prefixes);
			// ext.addGazeteerFeatures("per", pers);
			// ext.addGazeteerFeatures("org", orgs);
			// ext.addGazeteerFeatures("pol", pol);
			// ext.addGazeteerFeatures("eco", eco);
			ext.addTitleWordFeatures();
			// ext.addTopicFeatures(llw);

		}

		return ext;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		FakeApps fa = new FakeApps();
		fa.runM1();
		// fa.runM2();

		System.out.println("process ends.");
	}

	private NeuralNetParams nnp;

	private Indexer<String> labelIdxer;

	private LDocumentCollection C1;

	private LDocumentCollection C2;

	private DenseTensor X = new DenseTensor();

	private DenseMatrix Y = new DenseMatrix();

	private DenseTensor Xv = new DenseTensor();

	private DenseMatrix Yv = new DenseMatrix();

	private DenseTensor Xt = new DenseTensor();

	private DenseMatrix Yt = new DenseMatrix();

	public FakeApps() {
		nnp = getNeuralNetParams();

		labelIdxer = Generics.newIndexer();
		labelIdxer.add("non-fake");
		labelIdxer.add("fake");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			LToken.INDEXER = l;
		}
	}

	private void buildData() {
		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < C1.size(); i++) {
			LDocument d = C1.get(i);
			String label = d.getAttrMap().get("label");
			int l = labelIdxer.indexOf(label);

			M.add(l, i);
			N.add(l);
		}

		IntegerMatrix T = DataSplitter.splitGroupsByLabels(M, new double[] { 0.9, 0.1 });

		X = new DenseTensor();
		Y = new DenseMatrix();

		Xv = new DenseTensor();
		Yv = new DenseMatrix();

		Xt = new DenseTensor();
		Yt = new DenseMatrix();

		for (int i = 0; i < T.size(); i++) {
			IntegerArray L = T.get(i);
			for (int loc : L) {
				LDocument d = C1.get(loc);
				String label = d.getAttrMap().get("label");
				int lb = labelIdxer.indexOf(label);

				LSentence s = d.getTokens();

				DenseMatrix Xm = new DenseMatrix();
				Xm.ensureCapacity(s.size());

				for (LToken t : s) {
					Xm.add(t.getFeatureVector());
				}

				if (i == 0) {
					X.add(Xm);
					Y.add(new DenseVector(new int[] { lb }));
				} else {
					Xv.add(Xm);
					Yv.add(new DenseVector(new int[] { lb }));
				}
			}
		}

		for (LDocument d : C2) {
			String label = d.getAttrMap().get("label");

			LSentence s = d.getTokens();

			DenseMatrix Xm = new DenseMatrix();
			Xm.ensureCapacity(s.size());

			for (LToken t : s) {
				Xm.add(t.getFeatureVector());
			}

			Xt.add(Xm);
			int lb = labelIdxer.indexOf(label);
			if (lb >= 0) {
				Yt.add(new DenseVector(new int[] { lb }));
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xv.trimToSize();
		Yv.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();
	}

	private NeuralNetParams getNeuralNetParams() {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setWeightDecayL2(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setGradientClipCutoff(Double.MAX_VALUE);

		nnp.setThreadSize(6);
		nnp.setBatchSize(5);
		nnp.setGradientAccumulatorResetSize(1000);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.ADAM);

		return nnp;
	}

	private LDocumentCollection readExtraData(boolean is_m1) throws Exception {
		LDocumentCollection ret = new LDocumentCollection();
		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data_pos")) {
			String name = file.getName();

			if (is_m1) {
				if (name.startsWith("M1_train-3")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = "non-fake";
						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
			} else {
				if (name.startsWith("M2_train-3")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = "non-fake";
						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}

			}
		}

		return ret;
	}

	private LDocumentCollection readTestData(boolean is_M1) throws Exception {
		LDocumentCollection ret = new LDocumentCollection();
		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data_pos")) {
			String name = file.getName();

			if (is_M1) {
				if (name.startsWith("M1_train-3")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";

						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
			} else {
				if (name.startsWith("M2_train-3")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";

						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
			}
		}

		return ret;
	}

	private LDocumentCollection readTrainData(boolean is_m1_data) throws Exception {
		LDocumentCollection ret = new LDocumentCollection();
		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data_pos")) {
			String name = file.getName();

			if (is_m1_data) {
				if (name.startsWith("M1_train-1") || name.startsWith("M1_train-2")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";
						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						if (ps.get(3).length() > 0) {
							d.getAttrMap().put("cor_title", ps.get(3));
						}
						ret.add(d);
					}
				} else if (name.startsWith("M2_train-1")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = "non-fake";
						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
			} else {
				if (name.startsWith("M2_train-1") || name.startsWith("M2_train-2")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";
						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				} else if (name.startsWith("M1_train-1")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.split("\t", line);
						ps = StrUtils.unwrap(ps);

						String label = "non-fake";
						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
			}
		}

		return ret;
	}

	public void runM1() throws Exception {
		System.out.printf("run Mission-1\n");

		C1 = readTrainData(true);
		C2 = readTestData(true);

		C1.doPadding();
		C2.doPadding();

		String modelFileName = FNPath.DATA_DIR + "m1_model.ser";
		String extFileName = FNPath.DATA_DIR + "m1_ext.ser";

		NewsFeatureExtractor ext = getFeatureExtractor(null, C1);

		ext.extract(C1);

		ext.setIsTraining(false);

		ext.extract(C2);

		buildData();

		{
			System.out.printf("train size:\t%d\n", C1.size());
			Counter<String> c = Generics.newCounter();
			for (LDocument d : C1) {
				c.incrementCount(d.getAttrMap().get("label"), 1);
			}
			System.out.println(c.toString());
		}

		{
			System.out.printf("test size:\t%d\n", C2.size());
			Counter<String> c = Generics.newCounter();
			for (LDocument d : C2) {
				c.incrementCount(d.getAttrMap().get("label"), 1);
			}
			System.out.println(c.toString());
			System.out.println();
		}

		Indexer<String> wordIdxer = ext.getValueIndexers().get(0);

		int vocab_size = wordIdxer.size();
		int word_emb_size = 50;
		int feat_emb_size = 5;
		int output_size = 2;

		boolean read_nn_model = false;

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(), TaskType.SEQ_CLASSIFICATION);

		if (read_nn_model) {
			nn.readObject(modelFileName);
		} else {
			{
				ConcatenationLayer l = null;
				{
					Indexer<String> featIdxer = ext.getFeatureIndexer();
					List<Indexer<String>> valIdxers = ext.getValueIndexers();

					int feat_size = featIdxer.size();

					Map<Integer, Layer> L = Generics.newHashMap(featIdxer.size());

					for (int i = 0; i < feat_size; i++) {
						String feat = featIdxer.getObject(i);
						Indexer<String> valIdxer = valIdxers.get(i);
						int emb_size = feat_emb_size;

						if (feat.equals("word")) {
							emb_size = word_emb_size;
						} else {
							emb_size = feat_emb_size;
						}

						EmbeddingLayer ll = new EmbeddingLayer(valIdxer.size(), emb_size, true, i);
						L.put(i, ll);
					}

					l = new ConcatenationLayer(featIdxer.size(), L);

					nn.add(l);
				}
			}

			int num_filters = 100;
			int[] window_sizes = new int[] { 2, 3, 5 };

			nn.add(new MultiWindowConvolutionalLayer(nn.get(nn.size() - 1).getOutputSize(), window_sizes, num_filters));
			// nn.add(new ConvolutionalLayer(emb_size, 4, num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new MultiWindowMaxPoolingLayer(num_filters, window_sizes));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters * window_sizes.length, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepareTraining();
			nn.initWeights();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.setCopyBestModel(false);

			int max_iters = 50;

			boolean use_2K = true;

			if (use_2K) {
				IntegerArray L = new IntegerArray(Y.size());

				for (DenseVector Ym : Y) {
					L.add((int) Ym.value(0));
				}

				IntegerMatrix G = DataSplitter.groupByLabels(L);
				IntegerArray nonFakeLocs = G.get(0);
				IntegerArray fakeLocs = G.get(1);

				for (int i = 0, cnt = 0; i < max_iters; i++) {
					ArrayUtils.shuffle(nonFakeLocs.values());
					int fake_size = fakeLocs.size();

					for (int j = 0; j < nonFakeLocs.size();) {
						int k = Math.min(nonFakeLocs.size(), j + fake_size);
						IntegerArray locs = new IntegerArray(fake_size * 2);

						for (int l = j; l < k; l++) {
							locs.add(nonFakeLocs.get(l));
						}
						j = k;

						for (int loc : fakeLocs) {
							locs.add(loc);
						}

						locs.trimToSize();

						DenseTensor _X = X.subTensor(locs.values());
						DenseMatrix _Y = Y.subMatrix(locs.values());

						if (++cnt % 10 == 0) {
							trainer.train(_X, _Y, Xv, Yv, 1);
						} else {
							trainer.train(_X, _Y, null, null, 1);
						}

					}
				}
			} else {
				trainer.train(X, Y, Xv, Yv, max_iters);
			}

			if (Yt.size() > 0) {
				Performance p = trainer.evaluate(Xt, Yt);
				System.out.println(p);
				System.out.println();

			}

			trainer.finish();
		}

		nn.writeObject(modelFileName);
		ext.writeObject(extFileName);

		DenseTensor Yh = new DenseTensor();

		for (DenseMatrix Xm : Xt) {
			DenseTensor Ym = (DenseTensor) nn.forward(Xm.toDenseTensor());
			Yh.addAll(Ym);
		}

		write(FNPath.DATA_DIR + "M1_test_out.txt", Yh);
	}

	public void runM2() throws Exception {
		System.out.printf("run Mission-2\n");
		C1 = readTrainData(false);
		C2 = readTestData(false);

		// C1 = new DataGenerator(C1).generate();

		C1.doPadding();
		C2.doPadding();

		String modelFileName = FNPath.DATA_DIR + "m2_model.ser";
		String extFileName = FNPath.DATA_DIR + "m2_ext.ser";

		NewsFeatureExtractor ext = getFeatureExtractor(null, C1);

		ext.extract(C1);

		ext.setIsTraining(false);

		ext.extract(C2);

		buildData();

		{
			System.out.printf("train size:\t%d\n", C1.size());
			Counter<String> c = Generics.newCounter();
			for (LDocument d : C1) {
				c.incrementCount(d.getAttrMap().get("label"), 1);
			}
			System.out.println(c.toString());
		}

		System.out.printf("test size:\t%d\n", C2.size());
		System.out.println();

		Indexer<String> wordIdxer = ext.getValueIndexers().get(0);

		int vocab_size = wordIdxer.size();
		int word_emb_size = 50;
		int feat_emb_size = 10;
		int output_size = 2;

		boolean read_nn_model = false;

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(), TaskType.SEQ_CLASSIFICATION);

		if (read_nn_model) {
			nn.readObject(modelFileName);
		} else {
			{
				ConcatenationLayer l = null;
				{
					Indexer<String> featIdxer = ext.getFeatureIndexer();
					List<Indexer<String>> valIdxers = ext.getValueIndexers();

					int feat_size = featIdxer.size();

					Map<Integer, Layer> L = Generics.newHashMap(featIdxer.size());

					for (int i = 0; i < feat_size; i++) {
						String feat = featIdxer.getObject(i);
						Indexer<String> valIdxer = valIdxers.get(i);
						int emb_size = feat_emb_size;

						if (feat.equals("word")) {
							emb_size = word_emb_size;
						} else {
							emb_size = feat_emb_size;
						}

						EmbeddingLayer ll = new EmbeddingLayer(valIdxer.size(), emb_size, true, i);
						L.put(i, ll);
					}

					l = new ConcatenationLayer(featIdxer.size(), L);

					nn.add(l);
				}
			}

			int num_filters = 100;
			int[] window_sizes = new int[] { 5 };

			nn.add(new MultiWindowConvolutionalLayer(nn.get(nn.size() - 1).getOutputSize(), window_sizes, num_filters));
			// nn.add(new ConvolutionalLayer(emb_size, 4, num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new MultiWindowMaxPoolingLayer(num_filters, window_sizes));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters * window_sizes.length, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepareTraining();
			nn.initWeights();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.setCopyBestModel(false);

			int max_iters = 10;

			boolean use_2K = true;

			if (use_2K) {
				IntegerArray L = new IntegerArray(Y.size());

				for (DenseVector Ym : Y) {
					L.add((int) Ym.value(0));
				}

				IntegerMatrix G = DataSplitter.groupByLabels(L);
				IntegerArray nonFakeLocs = G.get(0);
				IntegerArray fakeLocs = G.get(1);

				for (int i = 0; i < max_iters; i++) {
					ArrayUtils.shuffle(nonFakeLocs.values());
					int fake_size = fakeLocs.size();

					for (int j = 0; j < nonFakeLocs.size();) {
						int k = Math.min(nonFakeLocs.size(), j + fake_size);
						IntegerArray locs = new IntegerArray(fake_size * 2);

						for (int l = j; l < k; l++) {
							locs.add(nonFakeLocs.get(l));
						}
						j = k;

						for (int loc : fakeLocs) {
							locs.add(loc);
						}

						locs.trimToSize();

						DenseTensor _X = X.subTensor(locs.values());
						DenseMatrix _Y = Y.subMatrix(locs.values());

						trainer.train(_X, _Y, Xv, Yv, 1);
					}
				}
			} else {
				trainer.train(X, Y, Xv, Yv, max_iters);
			}

			trainer.finish();
		}

		nn.writeObject(modelFileName);
		ext.writeObject(extFileName);

		DenseTensor Yh = new DenseTensor();

		for (DenseMatrix Xm : Xt) {
			DenseTensor Ym = (DenseTensor) nn.forward(Xm.toDenseTensor());
			Yh.addAll(Ym);
		}

		write(FNPath.DATA_DIR + "M2_test_out.txt", Yh);
	}

	private void write(String outFileName, DenseTensor Yt) {
		TextFileWriter writer = new TextFileWriter(outFileName);
		DecimalFormat df = new DecimalFormat("0.##########");

		for (int i = 0; i < Yt.size(); i++) {
			DenseVector yh = Yt.row(i).row(0);
			LDocument d = C2.get(i);
			String id = d.getAttrMap().get("id");
			writer.write(String.format("%s\t%s\n", id, df.format(yh.value(1))));
		}
		writer.close();

	}

}