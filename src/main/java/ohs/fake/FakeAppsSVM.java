package ohs.fake;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileWriter;
import ohs.ir.search.app.WordSearcher;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.eval.Performance;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterInitializer;
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

public class FakeAppsSVM {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		FakeAppsSVM fa = new FakeAppsSVM();
		// fa.runM1();
		fa.runM2();

		System.out.println("process ends.");
	}

	private NeuralNetParams nnp;

	private Indexer<String> labelIdxer;

	private LDocumentCollection C1;

	private LDocumentCollection C2;

	private DenseTensor X = new DenseTensor();

	private DenseTensor Y = new DenseTensor();

	private DenseTensor Xv = new DenseTensor();

	private DenseTensor Yv = new DenseTensor();

	private DenseTensor Xt = new DenseTensor();

	private DenseTensor Yt = new DenseTensor();

	private boolean is_m1 = true;

	public FakeAppsSVM() {
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

	private DenseTensor apply(NeuralNet nn, DenseTensor X) {
		System.out.println("apply to test docs.");
		DenseTensor Yh = new DenseTensor();

		for (DenseMatrix Xm : Xt) {
			DenseTensor Ym = (DenseTensor) nn.forward(Xm.toDenseTensor());
			Yh.addAll(Ym.copy(false));
		}

		return Yh;
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
		Y = new DenseTensor();

		Xv = new DenseTensor();
		Yv = new DenseTensor();

		Xt = new DenseTensor();
		Yt = new DenseTensor();

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
					Y.add(new DenseVector(new int[] { lb }).toDenseMatrix());
				} else {
					Xv.add(Xm);
					Yv.add(new DenseVector(new int[] { lb }).toDenseMatrix());
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xv.trimToSize();
		Yv.trimToSize();

		Xt = new DenseTensor();
		Yt = new DenseTensor();

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
				Yt.add(new DenseVector(new int[] { lb }).toDenseMatrix());
			}
		}

		Xt.trimToSize();
		Yt.trimToSize();
	}

	private void createCNN(NeuralNet nn, NewsFeatureExtractor ext, boolean is_m1, boolean use_pretrained)
			throws Exception {
		ConcatenationLayer l = null;

		int word_emb_size = 50;
		int feat_emb_size = 5;
		int output_size = 2;

		{
			Indexer<String> featIdxer = ext.getFeatureIndexer();
			List<Indexer<String>> valIdxers = ext.getValueIndexers();

			int feat_size = featIdxer.size();

			List<EmbeddingLayer> L = Generics.newArrayList(feat_size);

			for (int i = 0; i < feat_size; i++) {
				String feat = featIdxer.getObject(i);
				Indexer<String> valIdxer = valIdxers.get(i);
				int emb_size = feat_emb_size;

				if (feat.equals("word")) {
					emb_size = word_emb_size;
				} else {
					emb_size = feat_emb_size;
				}

				if (feat.equals("word") && use_pretrained) {
					DenseMatrix E = readPretrainedEmbeddings(new Vocab(valIdxer));
					EmbeddingLayer ll = new EmbeddingLayer(E, true, i);
					ll.setSkipInitWeights(true);

					L.add(ll);
					// L.add(new EmbeddingLayer(E.copy(), true, i));
					// L.add(new EmbeddingLayer(valIdxer.size(), emb_size, true, i));
				} else {
					EmbeddingLayer ll = new EmbeddingLayer(valIdxer.size(), emb_size, true, i);
					L.add(ll);
				}
			}

			l = new ConcatenationLayer(L);

			nn.add(l);
		}

		int num_filters = 100;
		int[] window_sizes = new int[] { 1 };
		int l2 = 20;

		if (is_m1) {
			window_sizes = new int[] { 2, 4 };
		} else {
			window_sizes = new int[] { 2, 4 };
		}

		nn.add(new MultiWindowConvolutionalLayer(nn.get(nn.size() - 1).getOutputSize(), window_sizes, num_filters));
		// nn.add(new ConvolutionalLayer(emb_size, 4, num_filters));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new MaxPoolingLayer(num_filters));
		nn.add(new MultiWindowMaxPoolingLayer(num_filters, window_sizes));
		nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(num_filters * window_sizes.length, l2));
		nn.add(new NonlinearityLayer(new ReLU()));
		nn.add(new FullyConnectedLayer(l2, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.createGradientHolders();
		nn.initWeights(new ParameterInitializer());
	}

	private Set<String> filter(Set<String> dict) {
		Set<String> ret = Generics.newHashSet(dict.size());
		for (String s : dict) {
			if (s.length() > 1) {
				ret.add(s);
			}
		}
		return ret;
	}

	private NewsFeatureExtractor getFeatureExtractor(String extFileName, LDocumentCollection X, boolean is_m1)
			throws Exception {
		NewsFeatureExtractor ext = new NewsFeatureExtractor();
		ext.setIsTraining(true);

		if (extFileName != null) {
			ext.readObject(extFileName);
		} else {
			Set<String> pers = Generics.newHashSet();
			Set<String> orgs = Generics.newHashSet();

			{
				String dirName = "../../data/fake_news/dict/";
				pers = FileUtils.readStringHashSetFromText(dirName + "pers.txt");
				orgs = FileUtils.readStringHashSetFromText(dirName + "orgs.txt");

				pers = filter(pers);
				orgs = filter(orgs);

			}

			Set<String> pol = Generics.newHashSet();
			Set<String> eco = Generics.newHashSet();
			Set<String> its = Generics.newHashSet();

			// {
			// String dirName = "../../data/fake_news/dict_topic2/";
			// Counter<String> c1 = FileUtils.readStringCounterFromText(dirName + "정치.txt");
			// Counter<String> c2 = FileUtils.readStringCounterFromText(dirName + "경제.txt");
			// Counter<String> c3 = FileUtils.readStringCounterFromText(dirName +
			// "IT-과학.txt");
			//
			// c1.keepTopNKeys(2000);
			// c2.keepTopNKeys(2000);
			// c3.keepTopNKeys(2000);
			//
			// pol = filter(c1.keySet());
			// eco = filter(c2.keySet());
			// its = filter(c3.keySet());
			// }

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

			if (is_m1) {
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
				ext.addWordSectionFeatures();
				ext.addTopicFeatures(llw);
			} else {
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
				// ext.addGazeteerFeatures("its", its);
				ext.addWordSectionFeatures();
				ext.addTopicFeatures(llw);
			}
		}

		return ext;
	}

	private NeuralNetParams getNeuralNetParams() {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(1);
		nnp.setLearnRateDecaySize(100);
		nnp.setL2WeightDecay(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setGradientClipCutoff(Double.MAX_VALUE);

		nnp.setThreadSize(6);
		nnp.setBatchSize(8);
		nnp.setGradientAccumulatorResetSize(1000);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.ADAM);

		return nnp;
	}

	private void readData() throws Exception {
		C1 = readTrainData(is_m1);
		C2 = readTestData(is_m1);

		Collections.shuffle(C1);

		showLabelCounts();

		C1.doPadding();
		C2.doPadding();
	}

	private DenseMatrix readPretrainedEmbeddings(Vocab nVocab) throws Exception {
		Vocab vocab = DocumentCollection.readVocab(FNPath.NAVER_NEWS_COL_DC_DIR + "vocab.ser");
		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(FNPath.NAVER_DATA_DIR + "/emb/glove_ra.ser");

		WordSearcher ws = new WordSearcher(vocab, E, null);

		DenseMatrix ret = new DenseMatrix(nVocab.size(), E.colSize());

		ParameterInitializer pi = new ParameterInitializer();

		pi.init(ret);

		for (int i = 0; i < nVocab.size(); i++) {
			String word = nVocab.getObject(i);
			DenseVector e = ws.getVector(word);

			if (e != null) {
				VectorUtils.copy(e, ret.row(i));
			}
		}
		ret.unwrapValues();

		return ret;
	}

	private LDocumentCollection readTestData(boolean is_M1) throws Exception {
		LDocumentCollection ret = new LDocumentCollection();
		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data_pos")) {
			String name = file.getName();

			if (is_M1) {
				if (name.startsWith("M1_test")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));

						String id = ps.get(0);
						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";

						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("id", id);
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
			} else {
				if (name.startsWith("M2_test")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));

						String id = ps.get(0);
						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";

						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("id", id);
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
				if (name.startsWith("M1_train-1") || name.startsWith("M1_train-2") || name.startsWith("M1_train-3")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));

						String id = ps.get(0);
						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";

						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("id", id);
						d.getAttrMap().put("label", label);

						if (ps.get(3).length() > 0) {
							d.getAttrMap().put("cor_title", ps.get(3));
						}
						ret.add(d);
					}
					// } else if (name.startsWith("M2_train-1") || name.startsWith("M2_train-2")
					// || name.startsWith("M2_train-3")) {
					// for (String line : FileUtils.readLinesFromText(file)) {
					// List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));
					//
					// String id = ps.get(0);
					// String label = "non-fake";
					//
					// LDocument d = LDocument.newDocument(ps.get(2));
					// d.getAttrMap().put("id", id);
					// d.getAttrMap().put("label", label);
					//
					// ret.add(d);
					// }
				}
			} else {
				if (name.startsWith("M2_train-1") || name.startsWith("M2_train-2")) {
					for (String line : FileUtils.readLinesFromText(file)) {
						List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));

						String id = ps.get(0);
						String label = ps.get(1);
						label = label.equals("0") ? "non-fake" : "fake";

						LDocument d = LDocument.newDocument(ps.get(2));
						d.getAttrMap().put("id", id);
						d.getAttrMap().put("label", label);

						ret.add(d);
					}
				}
				// else if (name.startsWith("M1_train-1")) {
				// for (String line : FileUtils.readLinesFromText(file)) {
				// List<String> ps = StrUtils.unwrap(StrUtils.split("\t", line));
				//
				// String id = ps.get(0);
				// String label = "non-fake";
				//
				// LDocument d = LDocument.newDocument(ps.get(2));
				// d.getAttrMap().put("id", id);
				// d.getAttrMap().put("label", label);
				//
				// ret.add(d);
				// }
				// }
			}
		}

		return ret;
	}

	public void runM1() throws Exception {
		System.out.printf("run Mission-1\n");

		is_m1 = true;
		boolean read_model = false;
		boolean train_model = true;
		boolean read_ext = false;
		boolean use_2K = true;
		int max_iters = 100;
		boolean use_pretrained = true;

		readData();

		String modelFileName = FNPath.DATA_DIR + "m1_model_svm.ser";
		String extFileName = FNPath.DATA_DIR + "m1_ext_svm.ser";

		NewsFeatureExtractor ext = new NewsFeatureExtractor();

		if (read_ext) {
			ext.readObject(extFileName);
			ext.setIsTraining(false);

			ext.extract(C1);
			ext.extract(C2);
		} else {
			ext = getFeatureExtractor(null, C1, is_m1);
			ext.extract(C1);
			ext.setIsTraining(false);
			ext.extract(C2);
		}

		buildData();

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(), TaskType.SEQ_CLASSIFICATION);

		if (read_model) {
			nn.readObject(modelFileName);
			nn.createGradientHolders();
		} else {
			createCNN(nn, ext, is_m1, use_pretrained);
		}

		if (train_model) {
			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.setCopyBestModel(false);

			if (use_2K) {
				IntegerArray L = new IntegerArray(Y.size());

				for (DenseMatrix Ym : Y) {
					L.add((int) Ym.value(0, 0));
				}

				IntegerMatrix G = DataSplitter.groupByLabels(L);
				IntegerArray nonFakeLocs = G.get(0);
				IntegerArray fakeLocs = G.get(1);
				int[][] ranges = BatchUtils.getBatchRanges(fakeLocs.size(), 200);

				for (int i = 0; i < max_iters; i++) {
					ArrayUtils.shuffle(nonFakeLocs.values());
					ArrayUtils.shuffle(fakeLocs.values());

					for (int j = 0; j < ranges.length; j++) {
						System.out.printf("[%d/%d, %d/%d]\n", i + 1, max_iters, j + 1, ranges.length);

						int[] r = ranges[j];
						int s = r[0];
						int e = r[1];
						int g = e - s;
						IntegerArray locs = new IntegerArray(g * 2);

						for (int k = s; k < e; k++) {
							locs.add(fakeLocs.get(k));
						}

						for (int k = 0; k < g; k++) {
							locs.add(nonFakeLocs.get(s + k));
						}

						DenseTensor _X = X.subTensor(locs.values());
						DenseTensor _Y = Y.subTensor(locs.values());

						trainer.train(_X, _Y, Xv, Yv, 1);
					}

				}
			} else {
				trainer.train(X, Y, Xv, Yv, max_iters);
			}

			trainer.finish();

			nn.writeObject(modelFileName);
			ext.writeObject(extFileName);
		}

		DenseTensor Yh = apply(nn, Xt);

		write(FNPath.DATA_DIR + "M1_test_out.txt", C2, Yh);
	}

	public void runM2() throws Exception {
		System.out.printf("run Mission-2\n");

		is_m1 = false;
		boolean read_model = false;
		boolean train_model = true;
		boolean read_ext = false;
		boolean use_2K = true;
		int max_iters = 100;
		boolean use_pretrained = false;

		readData();

		String modelFileName = FNPath.DATA_DIR + "m2_model.ser";
		String extFileName = FNPath.DATA_DIR + "m2_ext.ser";

		NewsFeatureExtractor ext = new NewsFeatureExtractor();

		if (read_ext) {
			ext.readObject(extFileName);
			ext.setIsTraining(false);

			ext.extract(C1);
			ext.extract(C2);
		} else {
			ext = getFeatureExtractor(null, C1, is_m1);
			ext.extract(C1);
			ext.setIsTraining(false);
			ext.extract(C2);
		}

		buildData();

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(), TaskType.SEQ_CLASSIFICATION);

		if (read_model) {
			nn.readObject(modelFileName);
			nn.createGradientHolders();
		} else {
			createCNN(nn, ext, is_m1, use_pretrained);
		}

		if (train_model) {
			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.setCopyBestModel(false);

			if (use_2K) {
				IntegerArray L = new IntegerArray(Y.size());

				for (DenseMatrix Ym : Y) {
					L.add((int) Ym.value(0, 0));
				}

				IntegerMatrix G = DataSplitter.groupByLabels(L);
				IntegerArray nonFakeLocs = G.get(0);
				IntegerArray fakeLocs = G.get(1);
				int[][] ranges = BatchUtils.getBatchRanges(fakeLocs.size(), 200);

				for (int i = 0; i < max_iters; i++) {
					ArrayUtils.shuffle(nonFakeLocs.values());
					ArrayUtils.shuffle(fakeLocs.values());

					for (int j = 0; j < ranges.length; j++) {
						System.out.printf("[%d/%d, %d/%d]\n", i + 1, max_iters, j + 1, ranges.length);

						int[] r = ranges[j];
						int s = r[0];
						int e = r[1];
						int g = e - s;
						IntegerArray locs = new IntegerArray(g * 2);

						for (int k = s; k < e; k++) {
							locs.add(fakeLocs.get(k));
						}

						for (int k = 0; k < g; k++) {
							locs.add(nonFakeLocs.get(s + k));
						}

						DenseTensor _X = X.subTensor(locs.values());
						DenseTensor _Y = Y.subTensor(locs.values());

						trainer.train(_X, _Y, Xv, Yv, 1);
					}

				}
			} else {
				trainer.train(X, Y, Xv, Yv, max_iters);
			}

			trainer.finish();

			nn.writeObject(modelFileName);
			ext.writeObject(extFileName);
		}

		DenseTensor Yh = apply(nn, Xt);

		write(FNPath.DATA_DIR + "M2_test_out.txt", C2, Yh);
	}

	private void showLabelCounts() {
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
	}

	private void write(String outFileName, LDocumentCollection C, DenseTensor Y) {
		System.out.printf("write at [%s]\n", outFileName);

		TextFileWriter writer = new TextFileWriter(outFileName);
		DecimalFormat df = new DecimalFormat("0.##########");

		for (int i = 0; i < Y.size(); i++) {
			DenseVector yh = Y.get(i).row(0);
			LDocument d = C.get(i);
			String id = d.getAttrMap().get("id");
			writer.write(String.format("%s\t%s\t%d\n", id, df.format(yh.value(1)), yh.argMax()));
		}
		writer.close();
	}

}