package ohs.fake;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.io.TextFileWriter;
import ohs.ir.search.app.WordSearcher;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
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
import ohs.types.generic.Vocab.SYM;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class FakeApps {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		runM1();
		// runM2();

		System.out.println("process ends.");
	}

	public static void expandData() {

	}

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

			// ext.addPosFeatures(poss);
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
			ext.addTopicFeatures(llw);

		}

		return ext;
	}

	public static void runM1() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(100);
		nnp.setWeightDecayL2(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setGradientClipCutoff(5);

		nnp.setThreadSize(5);
		nnp.setBatchSize(1);
		nnp.setGradientAccumulatorResetSize(1000);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.ADAM);

		Indexer<String> labelIdxer = Generics.newIndexer();
		labelIdxer.add("non-fake");
		labelIdxer.add("fake");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			LToken.INDEXER = l;
		}

		LDocumentCollection C1 = new LDocumentCollection();
		LDocumentCollection C2 = new LDocumentCollection();

		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data_pos")) {
			String fName = file.getName();
			if (fName.startsWith("M1_train_pos") || fName.startsWith("M1_train-2_pos")) {
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
					C1.add(d);
				}
			} else if (fName.startsWith("M2_train_pos")) {
				for (String line : FileUtils.readLinesFromText(file)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					String label = "non-fake";
					LDocument d = LDocument.newDocument(ps.get(2));
					d.getAttrMap().put("label", label);

					C1.add(d);
				}
			}
		}

		C1.doPadding();
		C2.doPadding();

		String modelFileName = FNPath.DATA_DIR + "m1_model.ser";
		String extFileName = FNPath.DATA_DIR + "m1_ext.ser";

		NewsFeatureExtractor ext = getFeatureExtractor(null, C1);

		ext.extract(C1);

		ext.setIsTraining(false);

		ext.extract(C2);

		// DataGenerator dg = new DataGenerator(train);
		// MCollection extData = dg.generate();
		//
		// train.addAll(extData);

		System.out.printf("train size:\t%d\n", C1.size());
		System.out.printf("test size:\t%d\n", C2.size());
		System.out.println();

		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < C1.size(); i++) {
			LDocument d = C1.get(i);
			String label = d.getAttrMap().get("label");
			int l = labelIdxer.indexOf(label);

			M.add(l, i);
			N.add(l);
		}

		// IntegerMatrix T = DataSplitter.splitGroups(M, new int[] { 150, 25 });

		IntegerMatrix T = DataSplitter.splitGroupsByLabels(M, new double[] { 0.9, 0.1 });

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		DenseTensor Xv = new DenseTensor();
		DenseMatrix Yv = new DenseMatrix();

		DenseTensor Xt = new DenseTensor();

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
		}

		X.trimToSize();
		Y.trimToSize();

		Xv.trimToSize();
		Yv.trimToSize();

		Xt.trimToSize();

		Indexer<String> wordIdxer = ext.getValueIndexers().get(0);

		int vocab_size = wordIdxer.size();
		int word_emb_size = 50;
		int feat_emb_size = 10;
		int output_size = 2;

		boolean read_nn_model = false;

		NeuralNet nn = new NeuralNet(labelIdxer, null, TaskType.SEQ_CLASSIFICATION);

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

			IntegerArray L = new IntegerArray(Y.size());

			for (DenseVector Ym : Y) {
				L.add((int) Ym.value(0));
			}

			IntegerMatrix G = DataSplitter.groupByLabels(L);
			IntegerArray nonFakeLocs = G.get(0);
			IntegerArray fakeLocs = G.get(1);

			IntegerArray large = fakeLocs;
			IntegerArray small = nonFakeLocs;

			if (small.size() > large.size()) {
				small = fakeLocs;
				large = nonFakeLocs;
			}

			int max_iters = 100;

			trainer.train(X, Y, Xv, Yv, max_iters);
			trainer.finish();
		}

		nn.writeObject(modelFileName);
		ext.writeObject(extFileName);

		{
			DenseTensor Yt = (DenseTensor) nn.forward(Xt);

			TextFileWriter writer = new TextFileWriter(FNPath.DATA_DIR + "M1_test_out.txt");
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

	public static void runM2() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();

		nnp.setLearnRate(0.001);
		nnp.setLearnRateDecay(0.9);
		nnp.setLearnRateDecaySize(200);
		nnp.setWeightDecayL2(1);
		nnp.setGradientDecay(1);
		nnp.setRegLambda(0.001);
		nnp.setGradientClipCutoff(5);

		nnp.setThreadSize(5);
		nnp.setBatchSize(5);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setK2(1);

		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setUseAverageGradients(false);
		nnp.setUseHardGradientClipping(false);

		nnp.setOptimizerType(OptimizerType.ADAM);

		Indexer<String> labelIdxer = Generics.newIndexer();
		labelIdxer.add("non-fake");
		labelIdxer.add("fake");

		{
			Indexer<String> l = Generics.newIndexer();
			l.add("word");
			l.add("pos");
			LToken.INDEXER = l;
		}

		String[] fileNames = { "M2_train_pos.txt", "M1_train_pos.txt", "M2_test_pos" };

		LDocumentCollection train = new LDocumentCollection();
		LDocumentCollection test = new LDocumentCollection();

		for (File file : FileUtils.getFilesUnder(FNPath.DATA_DIR + "data")) {
			if (file.getName().startsWith("M2_train_pos")) {
				for (String line : FileUtils.readLinesFromText(file)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					String label = ps.get(1);
					label = label.equals("0") ? "non-fake" : "fake";
					LDocument d = LDocument.newDocument(ps.get(2));
					d.getAttrMap().put("label", label);

					train.add(d);
				}
			} else if (file.getName().startsWith("M1_train_pos")) {
				for (String line : FileUtils.readLinesFromText(file)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);

					String label = "non-fake";
					LDocument d = LDocument.newDocument(ps.get(2));
					d.getAttrMap().put("label", label);

					train.add(d);
				}
			} else if (file.getName().startsWith("M2_test_pos")) {
				for (String line : FileUtils.readLinesFromText(file)) {
					List<String> ps = StrUtils.split("\t", line);
					ps = StrUtils.unwrap(ps);
					LDocument d = LDocument.newDocument(ps.get(2));
					d.getAttrMap().put("id", ps.get(0));
					test.add(d);
				}
			}
		}

		// DataGenerator dg = new DataGenerator(train);
		// MCollection extData = dg.generate();
		//
		// train.addAll(extData);

		System.out.printf("train size:\t%d\n", train.size());
		System.out.printf("test size:\t%d\n", test.size());
		System.out.println();

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());

		IntegerMatrix M = new IntegerMatrix();
		IntegerArray N = new IntegerArray();

		for (int i = 0; i < train.size(); i++) {
			LDocument d = train.get(i);
			String label = d.getAttrMap().get("label");
			int l = labelIdxer.indexOf(label);

			M.add(l, i);
			N.add(l);
		}

		// IntegerMatrix T = DataSplitter.splitGroups(M, new int[] { 150, 25 });

		IntegerMatrix T = DataSplitter.splitGroupsByLabels(M, new double[] { 0.9, 0.1 });

		DenseTensor X = new DenseTensor();
		DenseMatrix Y = new DenseMatrix();

		DenseTensor Xv = new DenseTensor();
		DenseMatrix Yv = new DenseMatrix();

		DenseTensor Xt = new DenseTensor();

		for (int loc : T.get(0)) {
			LDocument d = train.get(loc);
			for (String word : d.getTokens().getTokenStrings(0)) {
				vocab.add(word);
			}
		}

		for (int i = 0; i < T.size(); i++) {
			IntegerArray L = T.get(i);
			for (int loc : L) {
				LDocument d = train.get(loc);
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
					Xv.add(Xm);
					Yv.add(new DenseVector(new int[] { lb }));
				}
			}
		}

		for (LDocument d : test) {
			String label = d.getAttrMap().get("label");

			List<Integer> ws = vocab.indexesOf(d.getTokens().getTokenStrings(0), 0);

			DenseMatrix Xm = new DenseMatrix(ws.size(), 1);

			for (int j = 0; j < ws.size(); j++) {
				Xm.add(j, 0, ws.get(j));
			}

			Xt.add(Xm);
		}

		X.trimToSize();
		Y.trimToSize();

		Xv.trimToSize();
		Yv.trimToSize();

		Xt.trimToSize();

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		System.out.printf("data size: [%d -> %d]\n", size1, size2);
		System.out.printf("max len: [%d]\n", max_len);
		System.out.printf("min len: [%d]\n", min_len);

		int emb_size = 200;
		int output_size = 2;

		System.out.println(vocab.info());

		DenseMatrix E = new DenseMatrix();
		E.ensureCapacity(vocab.size());

		{
			String dir = FNPath.NAVER_DATA_DIR;
			String emdFileName = dir + "emb/glove_ra.ser";
			String vocabFileName = dir + "col/dc/vocab.ser";

			Vocab v = DocumentCollection.readVocab(vocabFileName);

			WordSearcher ws = new WordSearcher(v, new RandomAccessDenseMatrix(emdFileName, false), null);

			for (int i = 0; i < vocab.size(); i++) {
				String word = vocab.getObject(i);
				DenseVector e = ws.getVector(word);

				if (e == null) {
					e = new DenseVector(ws.getEmbeddingMatrix().colSize());
				}
				E.add(e);
			}
			// E.add(1d / VectorMath.normL2(E));
			emb_size = E.colSize();
		}

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_CLASSIFICATION);

		{

			int num_filters = 100;
			int[] window_sizes = new int[] { 2, 3, 4 };

			// nn.add(new EmbeddingLayer(vocab_size, emb_size, true));
			nn.add(new EmbeddingLayer(E, true, 0));

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

			IntegerArray L = new IntegerArray(Y.size());

			for (DenseVector Ym : Y) {
				L.add((int) Ym.value(0));
			}

			IntegerMatrix G = DataSplitter.groupByLabels(L);
			IntegerArray nonFakeLocs = G.get(0);
			IntegerArray fakeLocs = G.get(1);

			IntegerArray large = fakeLocs;
			IntegerArray small = nonFakeLocs;

			if (small.size() > large.size()) {
				small = fakeLocs;
				large = nonFakeLocs;
			}

			int max_iters = 100;

			// for (int i = 0; i < max_iters; i++) {
			// ArrayUtils.shuffle(large.values());
			//
			// for (int j = 0; j < large.size();) {
			// int k = Math.min(large.size(), j + small.size());
			// IntegerArray locs = new IntegerArray(small.size() * 2);
			//
			// for (int l = j; l < k; l++) {
			// locs.add(large.get(l));
			// }
			// j = k;
			//
			// locs.addAll(small);
			// locs.trimToSize();
			//
			// DenseTensor Xm = X.subTensor(locs.values());
			// DenseMatrix Ym = Y.subMatrix(locs.values());
			// trainer.train(Xm, Ym, Xv, Yv, 1);
			// }
			// }

			trainer.train(X, Y, Xv, Yv, max_iters);
			trainer.finish();
		}

		{
			DenseTensor Yt = (DenseTensor) nn.forward(Xt);

			TextFileWriter writer = new TextFileWriter(FNPath.DATA_DIR + "M2_test_out.txt");
			DecimalFormat df = new DecimalFormat("0.##########");

			for (int i = 0; i < Yt.size(); i++) {
				DenseVector yh = Yt.row(i).row(0);
				LDocument d = test.get(i);
				String id = d.getAttrMap().get("id");
				writer.write(String.format("%s\t%s\n", id, df.format(yh.value(1))));
			}
			writer.close();
		}

	}

}