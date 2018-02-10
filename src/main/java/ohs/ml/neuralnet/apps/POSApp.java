package ohs.ml.neuralnet.apps;

import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.com.TaskType;
import ohs.ml.neuralnet.com.TextFeatureExtractor;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.ConcatenationLayer;
import ohs.ml.neuralnet.layer.ConvNetLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer.Type;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class POSApp {
	public static TextFeatureExtractor getNERFeatureExtractor(String extFileName, LDocument C) throws Exception {
		TextFeatureExtractor ext = new TextFeatureExtractor();

		if (extFileName != null) {
			ext.readObject(extFileName);
		} else {
			String dirName = "../../data/ml_data/senna/hash/";

			Set<String> caps = FileUtils.readStringHashSetFromText(dirName + "caps.lst");
			Set<String> suffixes = FileUtils.readStringHashSetFromText(dirName + "suffix.lst");
			Set<String> pers = FileUtils.readStringHashSetFromText(dirName + "ner.per.lst");
			Set<String> orgs = FileUtils.readStringHashSetFromText(dirName + "ner.org.lst");
			Set<String> locs = FileUtils.readStringHashSetFromText(dirName + "ner.loc.lst");
			Set<String> miscs = FileUtils.readStringHashSetFromText(dirName + "ner.misc.lst");

			Set<String> poss = Generics.newHashSet(C.getTokens().getTokenStrings(1));
			Set<String> prefixes = Generics.newHashSet();

			{
				Counter<String> c = Generics.newCounter();
				for (String word : C.getTokens().getTokenStrings(0)) {
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
			ext.addCapitalFeatures();
			// ext.addPuctuationFeatures();
			ext.addShapeOneFeatures();
			ext.addShapeTwoFeatures();
			ext.addShapeThreeFeatures();
			ext.addVowelConsonantFeatures();
			// ext.addSuffixFeatures(suffixes);
			// ext.addPrefixFeatures(prefixes);
			ext.addBinaryFeatures("per", pers);
			ext.addBinaryFeatures("org", orgs);
			ext.addBinaryFeatures("loc", locs);
			ext.addBinaryFeatures("misc", miscs);
			// ext.addCharacterFeatures();
		}

		return ext;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		run();
		System.out.println("process ends.");
	}

	public static LDocumentCollection readData(String fileName) throws Exception {
		LDocument d = LDocument.newDocument(FileUtils.readFromText(fileName));
		LDocumentCollection ret = new LDocumentCollection(d.size());
		for (LSentence s : d) {
			ret.add(s.toDocument());
		}
		return ret;
	}

	public static void run() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(1);
		nnp.setLearnRateDecaySize(10);
		nnp.setL2WeightDecay(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.3);
		nnp.setGradientClipCutoff(5);

		nnp.setThreadSize(6);
		nnp.setBatchSize(4);
		nnp.setGradientAccumulatorResetSize(Integer.MAX_VALUE);

		nnp.setK1(10);
		nnp.setK2(10);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.ADAM);

		String trainFileName = "../../data/ml_data/conll2003.bio2/train.dat";
		String testFileName = "../../data/ml_data/conll2003.bio2/test.dat";

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			l.add("chunk");
			l.add("ner");
			LToken.INDEXER = l;
		}

		LDocumentCollection C1 = readData(trainFileName);
		LDocumentCollection C2 = readData(testFileName);

		C1.doPadding();
		C2.doPadding();

		Indexer<String> labelIdxer = Generics.newIndexer();

		boolean read_ext = false;

		String extFileName = "../../data/ml_data/pos_ext.ser";

		TextFeatureExtractor ext = new TextFeatureExtractor();

		if (read_ext) {
			ext.readObject(extFileName);
			ext.setIsTraining(false);

			ext.extract(C1);
			ext.extract(C2);
		} else {
			String dirName = "../../data/ml_data/senna/hash/";

			Set<String> suffixes = FileUtils.readStringHashSetFromText(dirName + "suffix.lst");

			Map<String, String> binaryValueFiles = Generics.newHashMap();
			binaryValueFiles.put("per", dirName + "ner.per.lst");
			binaryValueFiles.put("org", dirName + "ner.org.lst");
			binaryValueFiles.put("loc", dirName + "ner.loc.lst");
			binaryValueFiles.put("misc", dirName + "ner.misc.lst");

			ext = TextFeatureExtractor.getTextFeatureExtractor(C1, binaryValueFiles, null, suffixes);

			ext.extract(C1);
			ext.setIsTraining(false);
			ext.extract(C2);
		}

		DenseTensor X = new DenseTensor();
		DenseTensor Y = new DenseTensor();

		X.ensureCapacity(C1.size());
		Y.ensureCapacity(C1.size());

		DenseTensor Xt = new DenseTensor();
		DenseTensor Yt = new DenseTensor();

		Xt.ensureCapacity(C2.size());
		Yt.ensureCapacity(C2.size());

		{
			LDocumentCollection C = new LDocumentCollection();
			C.addAll(C1);
			C.addAll(C2);

			Set<String> labels = Generics.newTreeSet(C.getTokenStrings(1));
			labels.remove("O");

			labelIdxer.addAll(labels);
			labelIdxer.add("O");

			for (int i = 0; i < C.size(); i++) {
				LSentence s = C.get(i).get(0);

				DenseMatrix Xm = new DenseMatrix();
				DenseMatrix Ym = new DenseMatrix();

				Xm.ensureCapacity(s.size());
				Ym.ensureCapacity(s.size());

				for (int j = 0; j < s.size(); j++) {
					LToken t = s.get(j);

					Xm.add(t.getFeatureVector());
					Ym.add(new DenseVector(new double[] { labelIdxer.getIndex(t.getString(3)) }));
				}

				if (i < C1.size()) {
					X.add(Xm);
					Y.add(Ym);
				} else {
					Xt.add(Xm);
					Yt.add(Ym);
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		DenseMatrix T = new DenseMatrix(labelIdxer.size(), labelIdxer.size());

		for (DenseMatrix Ym : Y) {
			for (int i = 1; i < Ym.rowSize(); i++) {
				int s_prev = (int) Ym.value(i - 1, 0);
				int s = (int) Ym.value(i, 0);
				T.add(s_prev, s, 1);
			}
		}

		T.add(1);

		T.normalizeRows();

		Indexer<String> wVocab = ext.getValueIndexers().get(0);

		System.out.println(wVocab.info());
		System.out.println(labelIdxer.info());
		System.out.println(labelIdxer.toString());
		System.out.println(X.sizeOfEntries());

		int voc_size = wVocab.size();
		int word_emb_size = 50;
		int feat_emb_size = 5;
		int l1_size = 100;
		int l2_size = 30;
		int label_size = labelIdxer.size();
		int type = 2;

		int k1 = nnp.getK1();
		int k2 = nnp.getK2();

		boolean use_ext_embs = true;
		DenseMatrix E = null;

		if (use_ext_embs) {
			List<String> words = FileUtils.readLinesFromText("../../data/ml_data/senna/hash/words.lst");
			List<String> embs = FileUtils.readLinesFromText("../../data/ml_data/senna/embeddings/embeddings.txt");

			E = new DenseMatrix(wVocab.size(), 50);

			new ParameterInitializer().init(E);

			for (int i = 0; i < words.size(); i++) {
				String word = words.get(i);

				int w = wVocab.indexOf(word);

				if (w <= 0) {
					continue;
				}
				DenseVector e = E.row(w);
				e.setAll(0);

				String s = embs.get(i);
				String[] vs = s.split(" ");

				for (int j = 0; j < vs.length; j++) {
					e.add(j, Double.parseDouble(vs[j]));
				}
			}

			// VectorMath.unitVector(E, E);

		}

		String modelFileName = "../../data/ml_data/pos_model.ser";

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(wVocab), TaskType.TOKEN_CLASSIFICAITON);

		boolean read_ner_model = false;

		if (read_ner_model && FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			// EmbeddingLayer l = (EmbeddingLayer) nn.get(0);
			// l.setLearnEmbedding(false);
			nn.createGradientHolders();
		} else {
			ConcatenationLayer l1 = null;
			NeuralNet cnn = null;
			{
				Indexer<String> featIdxer = ext.getFeatureIndexer();
				List<Indexer<String>> valIdxers = ext.getValueIndexers();

				int feat_size = featIdxer.size();

				List<EmbeddingLayer> L = Generics.newArrayList(feat_size);

				for (int i = 0; i < feat_size; i++) {
					String feat = featIdxer.getObject(i);
					Indexer<String> valIdxer = valIdxers.get(i);
					int emb_size = feat_emb_size;

					if (feat.equals("ch")) {
						cnn = new NeuralNet();
						cnn.add(new EmbeddingLayer(valIdxer.size(), emb_size, true, 0));
						cnn.add(new ConvNetLayer(emb_size, 3, 25));
					} else {
						EmbeddingLayer ll = null;
						if (feat.equals("word") && E != null) {
							ll = new EmbeddingLayer(E, false, i);
							ll.setSkipInitWeights(true);
							L.add(ll);
							L.add(new EmbeddingLayer(valIdxer.size(), 30, true, i));
						} else {
							ll = new EmbeddingLayer(valIdxer.size(), emb_size, true, i);
							L.add(ll);
						}

					}
				}

				l1 = new ConcatenationLayer(L);

				nn.add(l1);
			}

			nn.add(new DropoutLayer());
			// nn.add(new BatchNormalizationLayer(l1.getOutputSize()));

			// nn.add(new ConvolutionalLayer(l.getOutputSize(), 3, 50));
			// nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new MaxPoolingLayer(50));
			// nn.add(new DropoutLayer());

			// nn.add(new RnnLayer(l.getOutputSize(), l1_size, bptt_size, new ReLU()));
			// nn.add(new LstmLayer(l.getOutputSize(), l1_size, bptt_size));
			nn.add(new BidirectionalRecurrentLayer(Type.LSTM, l1.getOutputSize(), l1_size, k1, k2, new ReLU()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new LayerNormalizationLayer(l1_size));
			// nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			// nn.add(new FullyConnectedLayer(l1_size, l2_size));
			// nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new DropoutLayer());
			// nn.add(new FullyConnectedLayer(l2_size, label_size));
			nn.add(new SoftmaxLayer(label_size));

			nn.setIsTraining(true);
			nn.createGradientHolders();

			ParameterInitializer pi = new ParameterInitializer();
			nn.initWeights(pi);

		}

		// nn.writeObject(modelFileName);

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);

		int max_iters = 30;
		boolean use_batches = true;

		if (use_batches) {
			IntegerArray locs = new IntegerArray(ArrayUtils.range(X.size()));
			int group_size = 2000;
			int[][] rs = BatchUtils.getBatchRanges(X.size(), group_size);

			for (int i = 0; i < max_iters; i++) {
				ArrayUtils.shuffle(locs.values());

				for (int j = 0; j < rs.length; j++) {
					System.out.printf("iters [%d/%d], batches [%d/%d]\n", i + 1, max_iters, j + 1, rs.length);

					int[] r = rs[j];
					int r_size = r[1] - r[0];
					DenseTensor Xm = new DenseTensor();
					DenseTensor Ym = new DenseTensor();

					Xm.ensureCapacity(r_size);
					Ym.ensureCapacity(r_size);

					for (int k = r[0]; k < r[1]; k++) {
						int loc = locs.get(k);
						Xm.add(X.get(loc));
						Ym.add(Y.get(loc));
					}

					trainer.train(Xm, Ym, Xt, Yt, 1);
				}
			}
		} else {
			trainer.train(X, Y, Xt, Yt, max_iters);
		}

		trainer.finish();

		nn.writeObject(modelFileName);
		ext.writeObject(extFileName);
	}

	public static void transformBIOtoBIOSE(LDocument d) {

		int w_loc = 0;
		int ne_loc = 3;

		for (int u = 0; u < d.size(); u++) {
			LSentence s = d.get(u);
			for (int i = 0; i < s.size();) {
				LToken t1 = s.get(i);
				String w1 = t1.getString(w_loc);
				String ne1 = t1.getString(ne_loc);

				if (ne1.startsWith("B-")) {
					int size = 1;

					for (int k = i + 1; k < s.size(); k++) {
						LToken t2 = s.get(k);
						String w2 = t2.getString(w_loc);
						String ne2 = t2.getString(ne_loc);

						if (ne2.startsWith("I")) {
							size++;
						} else {
							break;
						}
					}

					int j = i + size;

					for (int k = i; k < j; k++) {
						LToken t2 = s.get(k);
						String ne = t2.getString(ne_loc);

						if (size == 1) {
							ne = ne.replace("B-", "S-");
						} else {
							if (k == i) {
								ne = ne.replace("B-", "B-");
							} else if (k == j - 1) {
								ne = ne.replace("I-", "E-");
							} else {
								ne = ne.replace("I-", "I-");
							}
						}
						t2.set(ne_loc, ne);
					}

					i = j;

				} else {
					i++;
				}
			}
		}
	}

	public static void transformBIOtoBIOSE(LDocumentCollection C) {
		for (LDocument d : C) {
			transformBIOtoBIOSE(d);
		}
	}
}
