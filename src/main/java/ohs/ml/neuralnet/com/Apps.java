package ohs.ml.neuralnet.com;

import java.io.File;
import java.util.List;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.EnglishTokenizer;
import ohs.fake.FNPath;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.search.app.WordSearcher;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.ConcatenationLayer;
import ohs.ml.neuralnet.layer.ConvNetLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.MultiWindowConvolutionalLayer;
import ohs.ml.neuralnet.layer.MultiWindowMaxPoolingLayer;
import ohs.ml.neuralnet.layer.NNConcatenationLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer.Type;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.nlp.ling.types.MCollection;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class Apps {

	public static NERFeatureExtractor getNERFeatureExtractor(String extFileName, MDocument trainData) throws Exception {
		NERFeatureExtractor ext = new NERFeatureExtractor();

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

			Set<String> poss = Generics.newHashSet(trainData.getTokens().getTokenStrings(1));
			Set<String> prefixes = Generics.newHashSet();

			{
				Counter<String> c = Generics.newCounter();
				for (String word : trainData.getTokens().getTokenStrings(0)) {
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
			// ext.addShapeOneFeatures();
			// ext.addShapeTwoFeatures();
			// ext.addShapeThreeFeatures();
			// ext.addSuffixFeatures(suffixes);
			// ext.addPrefixFeatures(prefixes);
			ext.addGazeteerFeatures("per", pers);
			ext.addGazeteerFeatures("org", orgs);
			ext.addGazeteerFeatures("loc", locs);
			ext.addGazeteerFeatures("misc", miscs);
			ext.addCharacterFeatures();
		}

		return ext;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// testMNIST();
		// testCharRNN();
		testNER();

		// testSentenceClassification();
		// testDocumentClassification();

		System.out.println("process ends.");
	}

	public static void testCharRNN() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(25);
		nnp.setIsFullSequenceBatch(false);
		nnp.setIsRandomBatch(false);
		nnp.setGradientAccumulatorResetSize(100);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setWeightDecayL2(0.9999);
		nnp.setGradientDecay(1);
		nnp.setK2(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();
		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		List<String> lines = Generics.newLinkedList();

		EnglishTokenizer et = new EnglishTokenizer();

		for (String line : FileUtils.readFromText("../../data/ml_data/shakespeare_input.txt").split("\n\n")) {
			lines.add(StrUtils.join(" ", et.tokenize(line)));
		}

		lines = Generics.newArrayList(lines);

		lines = lines.subList(0, 1000);

		int test_size = 1000;
		// int train_size = lines.size() - test_size;
		int train_size = 10000;

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());
		vocab.add(SYM.START.getText());
		vocab.add(SYM.END.getText());

		{
			Set<String> set = Generics.newTreeSet();

			for (int i = 0; i < lines.size() && i < train_size; i++) {
				String s = lines.get(i);
				for (int j = 0; j < s.length(); j++) {
					set.add(s.charAt(j) + "");
				}
			}

			for (String s : set) {
				vocab.add(s);
			}
		}

		for (int i = 0; i < lines.size(); i++) {
			String s = lines.get(i);

			IntegerArray t = new IntegerArray(s.length() + 2);
			t.add(vocab.indexOf(SYM.START.getText()));

			for (int j = 0; j < s.length(); j++) {
				int w = vocab.indexOf(s.charAt(j) + "", 0);
				t.add(w);
			}

			t.add(vocab.indexOf(SYM.END.getText()));

			DenseMatrix Xm = new DenseMatrix(t.size() - 1, 1);
			DenseVector Ym = new DenseVector(t.size() - 1);

			for (int j = 0; j < t.size() - 1; j++) {
				Xm.add(j, 0, t.get(j));
				Ym.add(j, t.get(j + 1));
			}

			if (i < train_size) {
				X.add(Xm);
				Y.add(Ym);
			} else {
				Xt.add(Xm);
				Yt.add(Ym);
			}
		}

		Indexer<String> labelIdxer = Generics.newIndexer(vocab.getObjects());

		int voc_size = vocab.size();
		int emb_size = 100;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;
		int k1 = nnp.getK1();
		int k2 = nnp.getK2();

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_LABELING);

		String modelFileName = "../../data/ml_data/char-rnn.ser";

		if (FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			nn.prepareTraining();
		} else {

			// EmbeddingLayer l = new EmbeddingLayer(voc_size, emb_size, true);
			// l.setOutputWordIndexes(false);
			// nn.add(l);

			nn.add(new EmbeddingLayer(voc_size, emb_size, true, 0));
			nn.add(new DropoutLayer());
			// nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size,
			// bptt, new ReLU()));
			// nn.add(new LstmLayer(emb_size, l1_size));
			nn.add(new RnnLayer(emb_size, l1_size, k1, k2, new ReLU()));
			// nn.add(new DropoutLayer());
			// nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.prepareTraining();
			nn.initWeights();
		}

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);

		IntegerArray locs = new IntegerArray(ArrayUtils.range(X.size()));

		int group_size = 1000;
		int[][] rs = BatchUtils.getBatchRanges(X.size(), group_size);

		int max_iters = 1000;

		SentenceGenerator sg = new SentenceGenerator(nn, vocab);

		for (int u = 0; u < max_iters; u++) {
			for (int i = 0; i < rs.length; i++) {
				System.out.printf("iters [%d/%d], batches [%d/%d]\n", u + 1, max_iters, i + 1, rs.length);
				for (int j = 0; j < rs.length; j++) {
					int[] r = rs[j];
					int r_size = r[1] - r[0];
					DenseTensor Xm = new DenseTensor();
					DenseMatrix Ym = new DenseMatrix();

					Xm.ensureCapacity(r_size);
					Ym.ensureCapacity(r_size);

					for (int k = r[0]; k < r[1]; k++) {
						int loc = locs.get(k);
						Xm.add(X.get(loc));
						Ym.add(Y.get(loc));
					}

					trainer.train(Xm, Ym, null, null, 1);
				}
			}

			if (u % 10 == 0) {
				for (int j = 0; j < 10; j++) {
					String s = sg.generate(100);
					System.out.println(s);
				}
			}

		}

		trainer.finish();
	}

	public static void testDocumentClassification() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(5);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setWeightDecayL2(1);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setK2(1);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		Indexer<String> labelIdxer = Generics.newIndexer();
		labelIdxer.add("non-fake");
		labelIdxer.add("fake");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			MToken.INDEXER = l;
		}

		String[] fileNames = { "Mission1_sample_1차수정본.txt" };

		File inFile = new File(FNPath.DATA_DIR + "train_01", fileNames[0]);

		MCollection data = new MCollection();

		for (String line : FileUtils.readLinesFromText(inFile)) {
			List<String> ps = StrUtils.split("\t", line);
			ps = StrUtils.unwrap(ps);

			String label = ps.get(1);
			label = label.equals("0") ? "non-fake" : "fake";

			MDocument d = MDocument.newDocument(ps.get(2));
			d.getAttrMap().put("label", label);

			data.add(d);
		}

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());

		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < data.size(); i++) {
			MDocument d = data.get(i);
			String label = d.getAttrMap().get("label");
			int l = labelIdxer.indexOf(label);

			M.add(l, i);
			N.add(l);
		}

		IntegerMatrix T = DataSplitter.splitGroupsByLabels(M, new int[] { 150, 25 });

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();
		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		for (int loc : T.get(0)) {
			MDocument d = data.get(loc);

			for (String word : d.getTokens().getTokenStrings(0)) {
				vocab.add(word);
			}
		}

		for (int i = 0; i < T.size(); i++) {
			IntegerArray L = T.get(i);
			for (int loc : L) {
				MDocument d = data.get(loc);
				String label = d.getAttrMap().get("label");
				int lb = labelIdxer.indexOf(label);

				List<Integer> ws = vocab.indexesOf(d.getTokens().getTokenStrings(0), 0);

				DenseMatrix Xm = new DenseMatrix(ws.size(), 1);
				for (int j = 0; j < ws.size(); j++) {
					Xm.add(j, 0, ws.get(j));
				}

				if (i == 0) {
					X.add(Xm);
					Y.add(new DenseVector(new int[] { lb }));
				} else {
					Xt.add(Xm);
					Yt.add(new DenseVector(new int[] { lb }));
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		System.out.printf("data size: [%d -> %d]\n", size1, size2);
		System.out.printf("max len: [%d]\n", max_len);
		System.out.printf("min len: [%d]\n", min_len);

		int vocab_size = vocab.size();
		int emb_size = 200;
		int l1_size = 100;

		int l2_size = 20;
		int output_size = 2;
		int type = 0;

		System.out.println(vocab.info());

		DenseMatrix E = new DenseMatrix();

		{
			String dir = FNPath.NAVER_DATA_DIR;
			String emdFileName = dir + "emb/glove_ra.ser";
			String vocabFileName = dir + "col/dc/vocab.ser";

			Vocab v = DocumentCollection.readVocab(vocabFileName);

			WordSearcher ws = new WordSearcher(v, new RandomAccessDenseMatrix(emdFileName, false), null);

			List<DenseVector> es = Generics.newArrayList(vocab_size);

			for (int i = 0; i < vocab.size(); i++) {
				String word = vocab.getObject(i);
				DenseVector e = ws.getVector(word);

				if (e == null) {
					e = new DenseVector(ws.getEmbeddingMatrix().colSize());
				} else {
					es.add(e);
				}
			}

			E = new DenseMatrix(es);

			// E.add(1d / VectorMath.normL2(E));

			emb_size = E.colSize();
		}

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_CLASSIFICATION);

		if (type == 0) {
			int num_filters = 100;
			int[] window_sizes = new int[] { 2, 3, 5 };

			nn.add(new EmbeddingLayer(vocab_size, emb_size, false, 0));

			nn.add(new MultiWindowConvolutionalLayer(emb_size, window_sizes, num_filters));
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
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		}
	}

	public static void testMNIST() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setInputSize(100);
		nnp.setHiddenSize(50);
		nnp.setOutputSize(10);
		nnp.setBatchSize(50);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setGradientClipCutoff(20);
		nnp.setOptimizerType(OptimizerType.ADAM);

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		Indexer<String> labelIdxer = Generics.newIndexer();
		Vocab vocab = new Vocab();

		{

			Pair<SparseMatrix, IntegerArray> train = DataReader.readFromSvmFormat("../../data/ml_data/mnist.txt");
			Pair<SparseMatrix, IntegerArray> test = DataReader.readFromSvmFormat("../../data/ml_data/mnist.t.txt");

			DenseMatrix _X = train.getFirst().toDenseMatrix();
			IntegerArray _Y = train.getSecond();

			DenseMatrix _Xt = test.getFirst().toDenseMatrix(test.getFirst().rowSize(), _X.colSize());
			IntegerArray _Yt = test.getSecond();

			Set<String> labels = Generics.newTreeSet();
			for (int y : _Y) {
				labels.add(y + "");
			}

			labelIdxer = Generics.newIndexer(labels);

			for (int i = 0; i < _X.colSize(); i++) {
				vocab.add(i + "");
			}

			X.ensureCapacity(_X.rowSize());
			Y = new DenseMatrix(_X.rowSize(), 1);

			for (int i = 0; i < _X.rowSize(); i++) {
				X.add(_X.row(i).toDenseMatrix());
				Y.add(i, 0, _Y.get(i));
			}

			Xt.ensureCapacity(_Xt.rowSize());
			Yt = new DenseMatrix(_Xt.rowSize(), 1);

			for (int i = 0; i < _Xt.rowSize(); i++) {
				Xt.add(_Xt.row(i).toDenseMatrix());
				Yt.add(i, 0, _Yt.get(i));
			}
		}

		int vocab_size = vocab.size();
		int l1_size = 100;
		int l2_size = 25;
		int output_size = labelIdxer.size();

		NeuralNet nn = new NeuralNet(labelIdxer, null, TaskType.CLASSIFICATION);

		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new Tanh()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepareTraining();
		nn.initWeights();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}

	public static void testNER() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(1);
		nnp.setLearnRateDecaySize(50);
		nnp.setWeightDecayL2(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setGradientClipCutoff(5);

		nnp.setThreadSize(5);
		nnp.setBatchSize(3);
		nnp.setGradientAccumulatorResetSize(Integer.MAX_VALUE);

		nnp.setK1(7);
		nnp.setK2(7);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.NADAM);

		String trainFileName = "../../data/ml_data/conll2003.bio2/train.dat";
		String testFileName = "../../data/ml_data/conll2003.bio2/test.dat";

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			l.add("chunk");
			l.add("ner");
			MToken.INDEXER = l;
		}

		MDocument _X = MDocument.newDocument(FileUtils.readFromText(trainFileName));
		MDocument _Y = MDocument.newDocument(FileUtils.readFromText(testFileName));

		transformBIOtoBIOSE(_X);
		transformBIOtoBIOSE(_Y);

		Indexer<String> labelIdxer = Generics.newIndexer();

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		X.ensureCapacity(_X.size());
		Y.ensureCapacity(_X.size());

		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		Xt.ensureCapacity(_X.size());
		Xt.ensureCapacity(_X.size());

		boolean read_ext = false;

		String extFileName = "../../data/ml_data/ner_ext.ser";

		NERFeatureExtractor ext = getNERFeatureExtractor(read_ext ? extFileName : null, _X);

		{
			MDocument d = new MDocument();
			d.addAll(_X);
			d.addAll(_Y);

			d = d.toPaddedDocument();

			ext.extract(d.subDocument(0, _X.size()));
			ext.setIsTraining(false);

			ext.extract(d.subDocument(_X.size(), d.size()));

			Set<String> labels = Generics.newTreeSet();

			for (MToken t : d.getTokens()) {
				String word = t.getString(0);
				if (word.equals(MSentence.START) || word.equals(MSentence.END)) {
					t.set(3, "O");
				}
				String ne = t.getString(3);
				labels.add(ne);
			}

			labels.remove("O");
			labelIdxer.addAll(labels);
			labelIdxer.add("O");

			for (int i = 0; i < d.size(); i++) {
				MSentence s = d.get(i);

				DenseMatrix Xm = new DenseMatrix();
				DenseVector Ym = new DenseVector(s.size());

				Xm.ensureCapacity(s.size());

				for (int j = 0; j < s.size(); j++) {
					MToken t = s.get(j);

					Xm.add(t.getFeatureVector());
					Ym.add(j, labelIdxer.getIndex(t.getString(3)));
				}

				if (i < _X.size()) {
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

		Indexer<String> wVocab = ext.getValueIndexers().get(0);

		System.out.println(wVocab.info());
		System.out.println(labelIdxer.info());
		System.out.println(labelIdxer.toString());
		System.out.println(X.sizeOfEntries());

		int voc_size = wVocab.size();
		int word_emb_size = 50;
		int feat_emb_size = 5;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;

		int k1 = nnp.getK1();
		int k2 = nnp.getK2();

		boolean use_ext_embs = false;
		DenseMatrix E = null;

		if (use_ext_embs) {
			List<String> words = FileUtils.readLinesFromText("../../data/ml_data/senna/hash/words.lst");
			List<String> embs = FileUtils.readLinesFromText("../../data/ml_data/senna/embeddings/embeddings.txt");

			E = new DenseMatrix(wVocab.size(), 50);

			for (int i = 0; i < words.size(); i++) {
				String word = words.get(i);

				int w = wVocab.indexOf(word);

				if (w <= 0) {
					continue;
				}

				String emb = embs.get(i);
				DenseVector e = E.row(w);
				String[] vs = emb.split(" ");

				for (int j = 0; j < vs.length; j++) {
					e.add(j, Double.parseDouble(vs[j]));
				}
			}
		}

		String modelFileName = "../../data/ml_data/ner_nn.ser";

		NeuralNet nn = new NeuralNet(labelIdxer, new Vocab(wVocab), TaskType.SEQ_LABELING);

		boolean read_ner_model = false;

		if (read_ner_model && FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			// EmbeddingLayer l = (EmbeddingLayer) nn.get(0);
			// l.setLearnEmbedding(false);
			nn.prepareTraining();
		} else {

			// if (E == null) {
			// nn.add(new EmbeddingLayer(voc_size, word_emb_size, true));
			// } else {
			// nn.add(new EmbeddingLayer(E, true));
			// }

			// nn.add(new CharConvNetLayer(10, 5, 50, ext.getFeatureIndexer().size()));

			ConcatenationLayer l1 = null;
			NeuralNet cnn = null;
			{
				Indexer<String> featIdxer = ext.getFeatureIndexer();
				List<Indexer<String>> valIdxers = ext.getValueIndexers();

				int feat_size = featIdxer.size();

				List<Layer> ls = Generics.newArrayList();

				for (int i = 0; i < feat_size; i++) {
					String feat = featIdxer.getObject(i);
					Indexer<String> valIdxer = valIdxers.get(i);
					int emb_size = feat_emb_size;

					if (feat.equals("word")) {
						emb_size = word_emb_size;
					} else if (feat.equals("pos")) {
						emb_size = (int) (1d * word_emb_size / 2);
					} else {
						emb_size = feat_emb_size;
					}

					if (feat.equals("ch")) {
						cnn = new NeuralNet();
						cnn.add(new EmbeddingLayer(valIdxer.size(), emb_size, true, 0));
						cnn.add(new ConvNetLayer(emb_size, 3, 25));
					} else {
						EmbeddingLayer ll = new EmbeddingLayer(valIdxer.size(), emb_size, true, i);
						ll.setOutputWordIndexes(false);
						ls.add(ll);
					}

				}

				l1 = new ConcatenationLayer(ls);

				nn.add(l1);

				if (cnn != null) {
					List<NeuralNet> nns = Generics.newArrayList();
					nns.add(cnn);

					nn.add(new NNConcatenationLayer(nns));
				}

			}

			// DiscreteFeatureEmbeddingLayer l = new
			// DiscreteFeatureEmbeddingLayer(ext.getFeatureValueIndexer().size(),
			// ext.getFeatureIndexer().size(), feat_emb_size, word_emb_size, true);

			nn.add(new DropoutLayer());

			// nn.add(new ConvolutionalLayer(l.getOutputSize(), 3, 50));
			// nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new MaxPoolingLayer(50));
			// nn.add(new DropoutLayer());

			// nn.add(new RnnLayer(l.getOutputSize(), l1_size, bptt_size, new ReLU()));
			// nn.add(new LstmLayer(l.getOutputSize(), l1_size, bptt_size));
			nn.add(new BidirectionalRecurrentLayer(Type.LSTM, l1.getOutputSize(), l1_size, k1, k2, new ReLU()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			// nn.add(new LayerNormalization(l1_size));
			// nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.prepareTraining();
			nn.initWeights();
		}

		nn.writeObject(modelFileName);

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
					DenseMatrix Ym = new DenseMatrix();

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

	public static void testSentenceClassification() throws Exception {
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

		Indexer<String> labelIdxer = Generics.newIndexer();
		labelIdxer.add("pos");
		labelIdxer.add("neg");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			MToken.INDEXER = l;
		}

		MDocument posData = new MDocument();
		MDocument negData = new MDocument();

		for (String line : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.pos")) {
			line = StrUtils.normalizeSpaces(line);
			MSentence s = MSentence.newSentence(line.replace(" ", "\n"));
			posData.add(s);

			s.getAttrMap().put("por", "pos");
		}

		for (String line : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.neg")) {
			line = StrUtils.normalizeSpaces(line);
			MSentence s = MSentence.newSentence(line.replace(" ", "\n"));
			negData.add(s);

			s.getAttrMap().put("por", "neg");
		}

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());

		for (MToken t : posData.getTokens()) {
			vocab.add(t.getString(0));
		}

		MDocument data = new MDocument();
		data.addAll(posData);
		data.addAll(negData);

		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < data.size(); i++) {
			MSentence s = data.get(i);
			String por = s.getAttrMap().get("por");
			int label = labelIdxer.indexOf(por);

			M.add(label, i);
			N.add(label);
		}

		IntegerMatrix T = DataSplitter.splitGroupsByLabels(M, new int[] { 4000, 1500 });

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();
		DenseTensor Xt = new DenseTensor();
		DenseMatrix Yt = new DenseMatrix();

		for (int i = 0; i < T.size(); i++) {
			IntegerArray L = T.get(i);
			for (int loc : L) {
				MSentence s = data.get(loc);
				String por = s.getAttrMap().get("por");
				int label = labelIdxer.indexOf(por);

				List<Integer> ws = vocab.indexesOf(s.getTokenStrings(0), 0);

				DenseMatrix Xm = new DenseMatrix(ws.size(), 1);
				for (int j = 0; j < ws.size(); j++) {
					Xm.add(j, 0, ws.get(j));
				}

				if (i == 0) {
					X.add(Xm);
					Y.add(new DenseVector(new int[] { label }));
				} else {
					Xt.add(Xm);
					Yt.add(new DenseVector(new int[] { label }));
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		int vocab_size = vocab.size();
		int emb_size = 50;
		int output_size = labelIdxer.size();
		int type = 0;

		System.out.println(vocab.info());

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_CLASSIFICATION);

		if (type == 0) {
			int num_filters = 100;
			int[] window_sizes = new int[] { 3, 4, 5 };

			nn.add(new EmbeddingLayer(vocab_size, emb_size, true, 0));

			nn.add(new MultiWindowConvolutionalLayer(emb_size, window_sizes, num_filters));
			// nn.add(new ConvolutionalLayer(emb_size, window_sizes[0], num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			// nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new MultiWindowMaxPoolingLayer(num_filters, window_sizes));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters * window_sizes.length, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepareTraining();
			nn.initWeights();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		}
	}

	public static void transformBIOtoBIOSE(MDocument d) {

		int w_loc = 0;
		int ne_loc = 3;

		for (int u = 0; u < d.size(); u++) {
			MSentence s = d.get(u);
			for (int i = 0; i < s.size();) {
				MToken t1 = s.get(i);
				String w1 = t1.getString(w_loc);
				String ne1 = t1.getString(ne_loc);

				if (ne1.startsWith("B-")) {
					int size = 1;

					for (int k = i + 1; k < s.size(); k++) {
						MToken t2 = s.get(k);
						String w2 = t2.getString(w_loc);
						String ne2 = t2.getString(ne_loc);

						if (ne2.startsWith("I")) {
							size++;
						} else {
							break;
						}
					}

					int j = i + size;
					;

					for (int k = i; k < j; k++) {
						MToken t2 = s.get(k);
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
}