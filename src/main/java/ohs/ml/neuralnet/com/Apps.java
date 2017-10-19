package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.SparseMatrix;
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer.Type;
import ohs.ml.neuralnet.layer.ConvolutionalLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.MaxPoolingLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
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

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// testMNIST();
		testCharRNN();
		// testNER();

		// testSentenceClassification();

		System.out.println("process ends.");
	}

	public static void testCharRNN() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(10);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(false);
		nnp.setGradientAccumulatorResetSize(100);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setTruncatedBackPropagationThroughTime(10);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

		IntegerMatrix X = new IntegerMatrix();
		IntegerMatrix Y = new IntegerMatrix();
		IntegerMatrix Xt = new IntegerMatrix();
		IntegerMatrix Yt = new IntegerMatrix();

		List<String> lines = Generics.newLinkedList();

		for (String line : FileUtils.readFromText("../../data/ml_data/shakespeare_input.txt").split("\n\n")) {
			line = StrUtils.normalizeSpaces(line);
			if (line.length() > 0) {
				lines.add(line);
			}
		}

		lines = Generics.newArrayList(lines);

		int test_size = 1000;
		int train_size = lines.size() - test_size;

		Vocab vocab = new Vocab();
		vocab.add(SYM.UNK.getText());
		vocab.add(SYM.START.getText());
		vocab.add(SYM.END.getText());

		{
			Set<String> charSet = Generics.newTreeSet();

			for (int i = 0; i < lines.size() && i < train_size; i++) {
				String s = lines.get(i);
				for (int j = 0; j < s.length(); j++) {
					charSet.add(s.charAt(j) + "");
				}
			}

			for (String ch : charSet) {
				vocab.add(ch);
			}
		}

		for (int i = 0; i < lines.size(); i++) {
			String s = lines.get(i);

			IntegerArray t = new IntegerArray(s.length());
			t.add(vocab.indexOf(SYM.START.getText()));

			for (int j = 0; j < s.length(); j++) {
				t.add(vocab.indexOf(s.charAt(j) + "", 0));
			}

			t.add(vocab.indexOf(SYM.END.getText()));

			IntegerArray x = new IntegerArray(t.size() - 1);
			IntegerArray y = new IntegerArray(t.size() - 1);

			for (int j = 0; j < t.size() - 1; j++) {
				x.add(t.get(j));
				y.add(t.get(j + 1));
			}

			if (i < train_size) {
				X.add(x);
				Y.add(y);
			} else {
				Xt.add(x);
				Yt.add(y);
			}
		}

		Indexer<String> labelIdxer = Generics.newIndexer(vocab.getObjects());

		int voc_size = vocab.size();
		int emb_size = 50;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_LABELING);

		String modelFileName = "../../data/ml_data/char-rnn.ser";

		if (FileUtils.exists(modelFileName)) {
			nn = new NeuralNet(modelFileName);
			nn.prepare();
		} else {
			nn.add(new EmbeddingLayer(voc_size, emb_size, true));
			// nn.add(new DropoutLayer());
			nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size,
					nnp.getTruncatedBackPropagationThroughTime(), new ReLU()));
			// nn.add(new BatchNormalizationLayer(l1_size));
			nn.add(new FullyConnectedLayer(l1_size, label_size));
			nn.add(new SoftmaxLayer(label_size));
			nn.prepare();
			nn.init();
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
					IntegerMatrix Xm = new IntegerMatrix(r_size);
					IntegerMatrix Ym = new IntegerMatrix(r_size);

					for (int k = r[0]; k < r[1]; k++) {
						int loc = locs.get(k);
						Xm.add(X.get(loc));
						Ym.add(Y.get(loc));
					}

					trainer.train(Xm, Ym, null, null, 1);

					// for (Layer l : nn) {
					// if (l instanceof RecurrentLayer) {
					// RecurrentLayer n = (RecurrentLayer) l;
					// n.resetH0();
					// }
					// }
					String s = sg.generate(100);
					System.out.println(s);
				}
			}
		}

		trainer.finish();
	}

	public static void testMNIST() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setInputSize(100);
		nnp.setHiddenSize(50);
		nnp.setOutputSize(10);
		nnp.setBatchSize(Integer.MAX_VALUE);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(8);
		nnp.setGradientClipCutoff(5);
		nnp.setOptimizerType(OptimizerType.ADAM);

		Pair<SparseMatrix, IntegerArray> train = DataReader.readFromSvmFormat("../../data/ml_data/mnist.txt");
		Pair<SparseMatrix, IntegerArray> test = DataReader.readFromSvmFormat("../../data/ml_data/mnist.t.txt");

		DenseMatrix X = train.getFirst().toDenseMatrix();
		IntegerArray Y = train.getSecond();

		DenseMatrix Xt = test.getFirst().toDenseMatrix(test.getFirst().rowSize(), X.colSize());
		IntegerArray Yt = test.getSecond();

		Set<String> labels = Generics.newTreeSet();
		for (int y : Y) {
			labels.add(y + "");
		}

		Indexer<String> labelIdxer = Generics.newIndexer(labels);

		int vocab_size = X.colSize();
		int l1_size = 200;
		int l2_size = 50;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet(labelIdxer, null, TaskType.CLASSIFICATION);

		nn.add(new FullyConnectedLayer(vocab_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(new ReLU()));
		// nn.add(new DropoutLayer());
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
		trainer.train(X, Y, Xt, Yt, 10000);
		trainer.finish();
	}

	public static void testNER() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setBatchSize(5);
		nnp.setIsFullSequenceBatch(true);
		nnp.setIsRandomBatch(true);
		nnp.setGradientAccumulatorResetSize(1000);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.001);
		nnp.setThreadSize(5);
		nnp.setTruncatedBackPropagationThroughTime(10);
		nnp.setOptimizerType(OptimizerType.ADAM);
		nnp.setGradientClipCutoff(5);

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

		MDocument trainData = MDocument.newDocument(FileUtils.readFromText(trainFileName));
		MDocument testData = MDocument.newDocument(FileUtils.readFromText(testFileName));

		Indexer<String> labelIdxer = Generics.newIndexer();
		Vocab vocab = new Vocab();
		vocab.add(Vocab.SYM.UNK.getText());

		IntegerMatrix X = new IntegerMatrix();
		IntegerMatrix Y = new IntegerMatrix();

		IntegerMatrix Xt = new IntegerMatrix();
		IntegerMatrix Yt = new IntegerMatrix();

		{
			WordFeatureExtractor ext = new WordFeatureExtractor();

			MDocument D = new MDocument();
			D.addAll(trainData);
			D.addAll(testData);

			Set<String> labels = Generics.newTreeSet();

			for (MToken t : D.getTokens()) {
				String word = t.getString(0);
				String ner = t.getString(3);
				String[] ps = ner.split("-");
				if (ps.length == 2) {
					ner = String.format("%s-%s", ps[1], ps[0]);
				}
				t.set(3, ner);
				ext.extract(t);

				labels.add(ner);
			}

			labels.remove("O");
			labelIdxer.addAll(labels);
			labelIdxer.add("O");

			for (int i = 0; i < D.size(); i++) {
				MSentence s = D.get(i);

				if (i < trainData.size()) {
					X.add(new IntegerArray(vocab.getIndexes(s.getTokenStrings(4))));
					Y.add(new IntegerArray(labelIdxer.indexesOf(s.getTokenStrings(3), 0)));
				} else {
					Xt.add(new IntegerArray(vocab.indexesOf(s.getTokenStrings(4), 0)));
					Yt.add(new IntegerArray(labelIdxer.indexesOf(s.getTokenStrings(3), 0)));
				}
			}
		}

		X.trimToSize();
		Y.trimToSize();

		Xt.trimToSize();
		Yt.trimToSize();

		System.out.println(vocab.info());
		System.out.println(labelIdxer.info());
		System.out.println(X.sizeOfEntries());

		int voc_size = vocab.size();
		int emb_size = 50;
		int l1_size = 100;
		int label_size = labelIdxer.size();
		int type = 2;

		String modelFileName = "../../data/ml_data/ner_nn.ser";

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_LABELING);

		if (type == 0) {
		} else if (type == 2) {
			if (FileUtils.exists(modelFileName)) {
				nn = new NeuralNet(modelFileName);
				nn.prepare();
			} else {
				nn.add(new EmbeddingLayer(voc_size, emb_size, true));
				// nn.add(new FullyConnectedLayer(voc_size, emb_size));
				nn.add(new DropoutLayer());
				nn.add(new BidirectionalRecurrentLayer(Type.LSTM, emb_size, l1_size,
						nnp.getTruncatedBackPropagationThroughTime(), new ReLU()));
				// nn.add(new BatchNormalizationLayer(l1_size));
				nn.add(new FullyConnectedLayer(l1_size, label_size));
				nn.add(new SoftmaxLayer(label_size));
				nn.prepare();
				nn.init();
			}

			// nn.writeObject(modelFileName);

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);

			IntegerArray locs = new IntegerArray(ArrayUtils.range(X.size()));

			int group_size = 1000;
			int[][] rs = BatchUtils.getBatchRanges(X.size(), group_size);

			int max_iters = 1000;

			for (int u = 0; u < max_iters; u++) {
				ArrayUtils.shuffle(locs.values());

				for (int i = 0; i < rs.length; i++) {
					System.out.printf("iters [%d/%d], batches [%d/%d]\n", u + 1, max_iters, i + 1, rs.length);
					for (int j = 0; j < rs.length; j++) {
						int[] r = rs[j];
						int r_size = r[1] - r[0];
						IntegerMatrix Xm = new IntegerMatrix(r_size);
						IntegerMatrix Ym = new IntegerMatrix(r_size);

						for (int k = r[0]; k < r[1]; k++) {
							int loc = locs.get(k);
							Xm.add(X.get(loc));
							Ym.add(Y.get(loc));
						}

						trainer.train(Xm, Ym, Xt, Yt, 1);
					}
				}
			}

			trainer.finish();

			nn.writeObject(modelFileName);
		}
	}

	public static void testSentenceClassification() throws Exception {
		NeuralNetParams nnp = new NeuralNetParams();
		nnp.setInputSize(100);
		nnp.setHiddenSize(50);
		nnp.setOutputSize(10);
		nnp.setBatchSize(10);
		nnp.setLearnRate(0.001);
		nnp.setRegLambda(0.01);
		nnp.setThreadSize(5);
		nnp.setTruncatedBackPropagationThroughTime(10);

		IntegerMatrix X = new IntegerMatrix();
		IntegerArray Y = new IntegerArray();
		IntegerMatrix Xt = new IntegerMatrix();
		IntegerArray Yt = new IntegerArray();

		Vocab vocab = new Vocab();
		Indexer<String> labelIdxer = Generics.newIndexer();

		{
			List<String> sents = Generics.newLinkedList();

			for (String s : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.pos")) {
				sents.add(s + "\tpos");
			}

			for (String s : FileUtils.readLinesFromText("../../data/sentiment/rt-polarity.neg")) {
				sents.add(s + "\tneg");
			}

			IntegerMatrix M = new IntegerMatrix();
			IntegerArray N = new IntegerArray();

			for (int i = 0; i < sents.size(); i++) {
				String s = sents.get(i);
				String[] ps = s.split("\t");
				String sentiment = ps[1];
				int label = sentiment.equals("pos") ? 0 : 1;
				M.add(label, i);
				N.add(label);
			}

			IntegerMatrix T = DataSplitter.splitGroups(M, new int[] { 5000, 500 });

			for (int i = 0; i < T.size(); i++) {
				IntegerArray L = T.get(i);
				for (int loc : L) {
					String[] ps = sents.get(loc).split("\t");
					String sent = ps[0];
					String sentiment = ps[1];
					int label = N.get(loc);

					List<Integer> ws = vocab.getIndexes(StrUtils.split(sent));
					if (i == 0) {
						X.add(new IntegerArray(ws));
						Y.add(N.get(loc));
					} else {
						Xt.add(new IntegerArray(ws));
						Yt.add(N.get(loc));
					}
				}
			}
		}

		int size1 = 0;
		int size2 = 0;
		int max_len = 0;
		int min_len = Integer.MAX_VALUE;

		int pad_label = vocab.getIndex("<PAD>");

		System.out.printf("data size: [%d -> %d]\n", size1, size2);
		System.out.printf("max len: [%d]\n", max_len);
		System.out.printf("min len: [%d]\n", min_len);

		int vocab_size = vocab.size();
		int input_size = vocab_size;
		int emb_size = 200;
		int l1_size = 100;
		int l2_size = 20;
		int output_size = 2;
		int type = 0;

		System.out.println(vocab.info());

		NeuralNet nn = new NeuralNet(labelIdxer, vocab, TaskType.SEQ_CLASSIFICATION);

		if (type == 0) {
			int num_filters = 100;
			int window_size = 2;
			int[] window_sizes = new int[] { 2 };

			nn.add(new EmbeddingLayer(vocab_size, emb_size, true));

			// nn.add(new MultiWindowConvolutionalLayer(emb_size, window_sizes,
			// num_filters));
			nn.add(new ConvolutionalLayer(emb_size, window_size, num_filters));
			nn.add(new NonlinearityLayer(new ReLU()));
			nn.add(new MaxPoolingLayer(num_filters));
			nn.add(new DropoutLayer());
			nn.add(new FullyConnectedLayer(num_filters, output_size));
			nn.add(new SoftmaxLayer(output_size));
			nn.prepare();
			nn.init();

			NeuralNetTrainer trainer = new NeuralNetTrainer(nn, nnp);
			trainer.train(X, Y, Xt, Yt, 10000);
			trainer.finish();
		}

	}
}