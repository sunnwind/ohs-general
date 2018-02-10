package ohs.ml.neuralnet.apps;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
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
import ohs.ml.neuralnet.layer.MultiWindowConvolutionalLayer;
import ohs.ml.neuralnet.layer.MultiWindowMaxPoolingLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
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
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class SentimentApp {

	public static TextFeatureExtractor getTextFeatureExtractor(LDocumentCollection C,
			Map<String, String> binaryValueFiles, Set<String> prefixes, Set<String> suffixes) throws Exception {

		TextFeatureExtractor ext = new TextFeatureExtractor();
		Set<String> poss = null;

		if (LToken.INDEXER.contains("pos")) {
			poss = Generics.newHashSet(C.getTokens().getTokenStrings(1));
		}

		prefixes = Generics.newHashSet();

		if (prefixes == null) {
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

		if (suffixes == null) {
			Counter<String> c = Generics.newCounter();
			for (String word : C.getTokens().getTokenStrings(0)) {
				if (word.length() > 3) {
					String p = word.substring(word.length() - 3, word.length());
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
			suffixes = c.keySet();
		}

		// if (poss != null) {
		// ext.addPosFeatures(poss);
		// }
		//
		// ext.addCapitalFeatures();
//		ext.addShapeOneFeatures();
//		ext.addShapeTwoFeatures();
//		ext.addShapeThreeFeatures();
//		ext.addVowelConsonantFeatures();
//		ext.addSuffixFeatures(suffixes);
//		ext.addPrefixFeatures(prefixes);
		//
		// if (binaryValueFiles != null) {
		// for (String feat : binaryValueFiles.keySet()) {
		// String fileName = binaryValueFiles.get(feat);
		// Set<String> dict = FileUtils.readStringHashSetFromText(fileName);
		// ext.addBinaryFeatures(feat, dict);
		// }
		// }

		// {
		// Set<String> s1 = Generics.newHashSet();
		// Set<String> s2 = Generics.newHashSet();
		//
		// CounterMap<String, String> cm = FileUtils
		// .readStringCounterMapFromText("../../data/medical_ir/phrs/wiki_phrs.txt",
		// false);
		//
		// for (String title : cm.keySet()) {
		// if (cm.containsKey("wkt")) {
		// s1.add(title.toLowerCase());
		// } else {
		// s2.add(title.toLowerCase());
		// }
		// }
		//
		// ext.addBinaryFeatures("wkt", s1);
		// ext.addBinaryFeatures("wkl", s2);
		// }

		return ext;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		SentimentApp sa = new SentimentApp();
		sa.run1();
		// run2();
		System.out.println("process ends.");
	}

	private DenseTensor X = new DenseTensor();

	private DenseTensor Y = new DenseTensor();

	private DenseTensor Xt = new DenseTensor();

	private DenseTensor Yt = new DenseTensor();

	private LDocumentCollection C1 = new LDocumentCollection();

	private LDocumentCollection C2 = new LDocumentCollection();

	private Indexer<String> labelIdxer = Generics.newIndexer();

	private void buildData() {
		X.ensureCapacity(C1.size());
		Y.ensureCapacity(C1.size());

		Xt.ensureCapacity(C2.size());
		Yt.ensureCapacity(C2.size());

		{
			LDocumentCollection C = new LDocumentCollection();
			C.addAll(C1);
			C.addAll(C2);

			for (int i = 0; i < C.size(); i++) {
				LDocument d = C.get(i);
				LSentence s = d.get(0);

				DenseMatrix Xm = new DenseMatrix();
				DenseMatrix Ym = new DenseMatrix();

				Xm.ensureCapacity(s.size());
				Ym.ensureCapacity(d.size());

				for (int j = 0; j < s.size(); j++) {
					LToken t = s.get(j);
					Xm.add(t.getFeatureVector());
				}

				String label = C.get(i).getAttrMap().get("label");

				Ym.add(new DenseVector(new double[] { labelIdxer.getIndex(label) }));

				Xm.unwrapValues();
				Ym.unwrapValues();

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
	}

	private void readData() throws Exception {
		labelIdxer = Generics.newIndexer();
		labelIdxer.add("pos");
		labelIdxer.add("neg");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			LToken.INDEXER = l;
		}

		for (String line : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.pos")) {
			line = StrUtils.normalizeSpaces(line);
			LDocument s = LSentence.newSentence(line.replace(" ", "\n")).toDocument();
			s.getAttrMap().put("label", "pos");

			C1.add(s);
		}

		for (String line : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.neg")) {
			line = StrUtils.normalizeSpaces(line);
			LDocument s = LSentence.newSentence(line.replace(" ", "\n")).toDocument();
			s.getAttrMap().put("label", "neg");
			C2.add(s);
		}

		LDocumentCollection C = new LDocumentCollection();
		C.addAll(C1);
		C.addAll(C2);

		Collections.shuffle(C);

		IntegerArray L = new IntegerArray(C.size());

		for (LDocument d : C) {
			L.add(labelIdxer.indexOf(d.getAttrMap().get("label")));
		}

		IntegerMatrix G = DataSplitter.groupByLabels(L);

		IntegerMatrix T = DataSplitter.splitGroupsByLabels(G, new int[] { 500, 100000 });

		for (int i = 0; i < T.size(); i++) {
			if (i == 0) {
				C2 = C.subCollection(T.get(i).values());
			} else {
				C1 = C.subCollection(T.get(i).values());
			}
		}
	}

	public void run1() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(50);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);

		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(Double.MAX_VALUE);

		readData();

		boolean read_ext = false;

		String extFileName = "../../data/ml_data/sm_ext.ser";

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

			ext = getTextFeatureExtractor(C1, null, null, suffixes);

			ext.extract(C1);
			ext.setIsTraining(false);
			ext.extract(C2);
		}

		buildData();

		int word_emb_size = 50;
		int feat_emb_size = 5;
		int label_size = labelIdxer.size();

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(), TaskType.SEQ_CLASSIFICATION);

		int num_filters = 100;
		int[] window_sizes = new int[] { 3, 5 };

		{
			Indexer<String> featIdxer = ext.getFeatureIndexer();
			List<Indexer<String>> valIdxers = ext.getValueIndexers();

			int feat_size = featIdxer.size();
			List<EmbeddingLayer> L = Generics.newArrayList(feat_size);

			for (int i = 0; i < feat_size; i++) {
				String feat = featIdxer.getObject(i);
				Indexer<String> valIdxer = valIdxers.get(i);
				int emb_size = feat.equals("word") ? word_emb_size : feat_emb_size;
				L.add(new EmbeddingLayer(valIdxer.size(), emb_size, true, i));
			}

			nn.add(new ConcatenationLayer(L));
		}

		nn.add(new MultiWindowConvolutionalLayer(nn.getLast().getOutputSize(), window_sizes, num_filters));
		// nn.add(new ConvolutionalLayer(emb_size, window_sizes[0], num_filters));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new MaxPoolingLayer(num_filters));
		nn.add(new MultiWindowMaxPoolingLayer(num_filters, window_sizes));
		nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(num_filters * window_sizes.length, label_size));
		nn.add(new SoftmaxLayer(label_size));

		nn.createGradientHolders();
		nn.initWeights(new ParameterInitializer());

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 1000);
		trainer.finish();
	}

	public void run2() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(5);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setK2(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		readData();

		boolean read_ext = false;

		String extFileName = "../../data/ml_data/sm_ext.ser";

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

		buildData();

		int vocab_size = ext.getValueIndexers().get(0).size();
		int word_emb_size = 50;
		int feat_emb_size = 5;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int k1 = 10;
		int k2 = 10;

		String modelFileName = "../../data/ml_data/ner_model.ser";

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(), TaskType.TOKEN_SEQ_LABELING);

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
					L.add(new EmbeddingLayer(valIdxer.size(), emb_size, true, i));
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
			nn.add(new DropoutLayer());
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

		nn.createGradientHolders();
		nn.initWeights(new ParameterInitializer());

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}
}
