package ohs.eden.keyphrase.mine;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.KoreanPosTokenizer;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.MemoryInvertedIndex;
import ohs.ir.search.index.PostingList;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.types.generic.Counter;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.StrUtils;

public class KorPhraseClassification {

	// public static String dir = MIRPath.TREC_CDS_2016_DIR;

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		KorPhraseClassification qc = new KorPhraseClassification();
		// qc.generateData();
		qc.vectorizeData();
		qc.trainClassifier();
		//

		qc.test();

		System.out.println("process ends.");
	}

	public void test() {
		System.out.println(Integer.MAX_VALUE);
		System.out.println(Short.MAX_VALUE);
	}

	public String delim = "  ";

	private DenseVector extractFeatures(RandomAccessDenseMatrix E, Vocab vocab, String kwd) throws Exception {
		IntegerArray Q = vocab.indexesOf(kwd.split(" "));

		StringBuffer sb = new StringBuffer();

		for (int w : Q) {
			String word = vocab.getObject(w);
			sb.append(word + " ");
		}

		String s = sb.toString().trim();

		int unseen_cnt = 0;

		for (int w : Q) {
			if (w < 0) {
				unseen_cnt++;
			}
		}

		if (unseen_cnt > 0) {
			return null;
		}

		DenseMatrix E2 = E.rowsAsMatrix(Q.values()).copy();

		Map<Integer, DenseVector> set = Generics.newHashMap();

		for (int i = 0; i < Q.size(); i++) {
			set.put(Q.get(i), E2.get(i));
		}

		DenseVector ne1 = new DenseVector(E.colSize());
		DenseVector ne2 = new DenseVector(E.colSize());

		for (int i = 0; i < E2.rowSize(); i++) {
			DenseVector e = E2.row(i);
			VectorMath.add(e, ne1);

			if (i == 0 || i < E2.rowSize() - 1) {
				VectorMath.add(e, ne2);
			}
		}

		ne1.multiply(1f / Q.size());
		ne2.multiply(1f / 2);

		DenseMatrix ret = new DenseMatrix(new DenseVector[] { ne1, ne1 });
		return ret.toDenseVector();
	}

	public void generateData() throws Exception {
		List<String> ins = FileUtils.readLinesFromText(KPPath.KYP_DIR + "ext/label_data.txt");
		Vocab vocab = DocumentCollection.readVocab(KPPath.COL_DC_DIR + "vocab.ser");
		KoreanPosTokenizer kpt = new KoreanPosTokenizer();
		// InvertedIndex ii = getInvertedIndex(lines, vocab, kpt);

		// ListMap<String, Integer> kwdToDocs = Generics.newListMap();

		List<Set<String>> docToKwds = Generics.newArrayList(ins.size());

		for (int i = 0; i < ins.size(); i++) {
			String line = ins.get(i);
			List<String> ps = Generics.newArrayList(line.split("\t"));
			ps = StrUtils.unwrap(ps);
			String kwdStr = ps.get(0);
			String title = ps.get(1);
			String abs = ps.get(2);

			Set<String> kwds = Generics.newHashSet();

			for (String kwd : kwdStr.split(StrUtils.LINE_REP)) {
				MSentence sent = MSentence.newSentence(kwd);

				for (Token t : sent.getTokens()) {
					String word = t.get(TokenAttr.WORD);
					t.set(TokenAttr.WORD.ordinal(), word.toLowerCase());
				}

				kwd = sent.toString().replace(" / ", "/").replace(" __ ", " ").replace(" + ", " ");
				kwds.add(kwd);
			}
			docToKwds.add(kwds);
		}

		Counter<String> kwdCnts = Generics.newCounter();

		{
			for (Set<String> l : docToKwds) {
				for (String k : l) {
					kwdCnts.incrementCount(k, 1);
				}
			}
		}

		kwdCnts.pruneKeysBelowThreshold(5);

		// Trie<String> ret = new Trie<String>();
		// for (String kwd : kwdCnts.keySet()) {
		// List<String> words = StrUtils.split(kwd);
		// Node<String> node = ret.insert(words);
		// node.setFlag(true);
		// }
		// ret.trimToSize();

		PhraseMapper<String> pm = new PhraseMapper<String>(PhraseMapper.createDict(kwdCnts.keySet()));

		List<String> outs = Generics.newArrayList(ins.size());

		for (int i = 0; i < ins.size(); i++) {
			String line = ins.get(i);
			List<String> ps = Generics.newArrayList(line.split("\t"));
			ps = StrUtils.unwrap(ps);
			String kwdStr = ps.get(0);
			String title = ps.get(1);
			String abs = ps.get(2);

			MDocument doc = MDocument.newDocument(title + "\n" + abs.replace(StrUtils.LINE_REP, "\n"));

			List<String> words = Generics.newArrayList(doc.sizeOfTokens());

			for (List<Token> ts : doc.getTokens()) {
				for (Token t : ts) {
					String word = t.get(TokenAttr.WORD);
					t.set(TokenAttr.WORD.ordinal(), word.toLowerCase());
					words.add(t.toString().replace(" / ", "/"));
				}
				words.add("\n");
			}

			Set<String> ansKwds = docToKwds.get(i);
			Set<String> posKwds = Generics.newHashSet();
			Set<String> negKwds = Generics.newHashSet();

			for (Pair<Integer, Integer> p : pm.map(words)) {
				String s = StrUtils.join(" ", words, p.getFirst(), p.getSecond());

				if (ansKwds.contains(s)) {
					posKwds.add(s);
				} else {
					negKwds.add(s);
				}
			}

			if (posKwds.size() == 0 || negKwds.size() == 0) {
				continue;
			}

			String s1 = StrUtils.join(StrUtils.LINE_REP, posKwds);
			String s2 = StrUtils.join(StrUtils.LINE_REP, negKwds);
			String s3 = StrUtils.normalizeSpaces(StrUtils.join(" ", words));
			s3 = s3.replace("\n", StrUtils.LINE_REP);

			String[] s = { s1, s2, s3 };
			s = StrUtils.wrap(s);

			outs.add(StrUtils.join("\t", s));
		}

		FileUtils.writeStringCollectionAsText(KPPath.KYP_DIR + "ext/label_data_2.txt", outs);
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

	private Map<Integer, DenseVector> getEmbeddings(Set<Integer> toks, RandomAccessDenseMatrix E) throws Exception {
		Map<Integer, DenseVector> ret = Generics.newHashMap(toks.size());

		for (int t : toks) {
			DenseVector e = new DenseVector(E.colSize());
			if (t != -1) {
				e = E.row(t);
			}
			ret.put(t, e);
		}
		return ret;
	}

	private InvertedIndex getInvertedIndex(List<String> lines, Vocab vocab, KoreanPosTokenizer kpt) {

		ListMapMap<Integer, Integer, Integer> lmm = Generics.newListMapMap(vocab.size(), ListType.LINKED_LIST);

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			List<String> ps = Generics.newArrayList(line.split("\t"));
			ps = StrUtils.unwrap(ps);
			String kwdStr = ps.get(0);
			String title = ps.get(1);
			String abs = ps.get(2);

			for (String kwd : kwdStr.split(StrUtils.LINE_REP)) {
				kwd = StrUtils.join(" ", kpt.tokenize(kwd));
				MSentence k = MSentence.newSentence(kwd);
				int j = 0;
				for (Token t : k.getTokens()) {
					int w = vocab.indexOf(t.toString());
					if (w != -1) {
						lmm.put(w, i, j++);
					}
				}
			}
		}

		Map<Integer, PostingList> m = Generics.newHashMap(lmm.size());

		for (int w : lmm.keySet()) {
			ListMap<Integer, Integer> lm = lmm.get(w);

			IntegerArray dseqs = new IntegerArray(lm.size());
			IntegerArrayMatrix posData = new IntegerArrayMatrix(lm.size());

			for (int dseq : lm.keySet()) {
				dseqs.add(dseq);
				posData.add(new IntegerArray(lm.get(dseq)));
			}

			PostingList pl = new PostingList(w, dseqs, posData);

			m.put(w, pl);
		}

		return new MemoryInvertedIndex(m, lines.size(), vocab);
	}

	public void trainClassifier() throws Exception {
		DenseMatrix XD = new DenseMatrix(KPPath.KYP_DIR + "ext/X.ser.gz");
		IntegerArray YD = new IntegerArray(KPPath.KYP_DIR + "ext/Y.ser.gz");

		DenseMatrix X = new DenseMatrix();
		IntegerArray Y = new IntegerArray();

		DenseMatrix Xt = new DenseMatrix();
		IntegerArray Yt = new IntegerArray();

		int test_size = 10000;

		{
			IntegerArrayMatrix G = DataSplitter.group(YD);

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
		param.setThreadSize(5);

		int feat_size = X.colSize();
		int l1_size = 100;
		int l2_size = 10;
		int output_size = labels.size();

		NeuralNet nn = new NeuralNet();
		nn.add(new FullyConnectedLayer(feat_size, l1_size));
		// nn.add(new BatchNormalizationLayer(l1_size));
		nn.add(new NonlinearityLayer(l1_size, new ReLU()));
		// nn.add(new DropoutLayer(l1_size));
		nn.add(new FullyConnectedLayer(l1_size, l2_size));
		// nn.add(new BatchNormalizationLayer(l2_size));
		nn.add(new NonlinearityLayer(l2_size, new ReLU()));
		// nn.add(new DropoutLayer(l2_size));
		nn.add(new FullyConnectedLayer(l2_size, output_size));
		nn.add(new SoftmaxLayer(output_size));
		nn.prepare();
		nn.init();

		NeuralNetTrainer trainer = new NeuralNetTrainer(nn, param, XD.rowSize(), null);

		IntegerArrayMatrix G = DataSplitter.group(Y);
		IntegerArray negLocs = G.get(0);
		IntegerArray posLocs = G.get(1);

		boolean stop = false;
		int max_iters = 1000;

		for (int i = 0; i < max_iters; i++) {
			ArrayUtils.shuffle(negLocs.values());
			int pos_size = posLocs.size();

			for (int j = 0; j < negLocs.size();) {
				int k = Math.min(negLocs.size(), j + pos_size);
				IntegerArray locs = new IntegerArray(pos_size * 2);

				for (int l = j; l < k; l++) {
					locs.add(negLocs.get(l));
				}
				j = k;

				locs.addAll(posLocs);
				locs.trimToSize();

				DenseMatrix Xs = X.rowsAsMatrix(locs.values());
				IntegerArray Ys = Y.subArray(locs.values());
				trainer.train(Xs, Ys, Xt, Yt, 1);
			}
		}

		trainer.finish();

		nn.writeObject(KPPath.KYP_DIR + "ext/nn_model.ser.gz");
	}

	public void vectorizeData() throws Exception {
		// DocumentCollection dc = new DocumentCollection(KPPath.COL_DC_DIR);
		// Vocab vocab = dc.getVocab();
		Vocab vocab = DocumentCollection.readVocab(KPPath.COL_DC_DIR + "vocab.ser");

		RandomAccessDenseMatrix E = new RandomAccessDenseMatrix(KPPath.COL_DIR + "emb/glove_emb_ra.ser", false);

		List<String> ins = FileUtils.readLinesFromText(KPPath.KYP_DIR + "ext/label_data_2.txt");

		List<DenseVector> X = Generics.newLinkedList();
		List<Integer> Y = Generics.newLinkedList();

		int doc_cnt = 0;
		for (String line : ins) {
			if (doc_cnt++ == 10000) {
				break;
			}

			String[] ps = line.split("\t");
			ps = StrUtils.unwrap(ps);

			List<String> posKwds = StrUtils.split(StrUtils.LINE_REP, ps[0]);
			List<String> negKwds = StrUtils.split(StrUtils.LINE_REP, ps[1]);
			List<String> contents = StrUtils.split(StrUtils.LINE_REP, ps[2]);

			DenseVector ed = new DenseVector(E.colSize());
			IntegerArray d = new IntegerArray();
			SparseVector dv1 = null;
			SparseVector dv2 = null;

			{
				Counter<Integer> c = Generics.newCounter();
				for (int i = 0; i < contents.size(); i++) {
					for (String tok : StrUtils.split(contents.get(i))) {
						int w = vocab.indexOf(tok.replace("/", " / "));
						if (w != -1) {
							c.incrementCount(w, 1);
						}
						d.add(w);
					}
				}
				d.trimToSize();
				dv1 = new SparseVector(c);
				dv2 = TermWeighting.tfidf(dv1, vocab);
			}

			{
				Set<Integer> toks = Generics.newHashSet();
				for (int t : dv1.indexes()) {
					String tok = vocab.getObject(t);
					if (tok.contains("/ NN")) {
						toks.add(t);
					}
				}

				Map<Integer, DenseVector> m = getEmbeddings(toks, E);
				double cnt_sum = 0;
				for (Entry<Integer, DenseVector> e : m.entrySet()) {
					int t = e.getKey();
					DenseVector et = e.getValue();
					double cnt = dv1.value(t);
					VectorMath.addAfterMultiply(et, cnt, ed);
					cnt_sum += cnt;
				}
				ed.multiply(1d / cnt_sum);
			}

			List<String>[] kwdsList = new List[] { posKwds, negKwds };

			for (int i = 0; i < kwdsList.length; i++) {
				List<String> kwds = kwdsList[i];
				for (String kwd : kwds) {
					IntegerArray k = new IntegerArray();
					SparseVector qv1 = null;
					SparseVector qv2 = null;

					{
						Counter<Integer> c = Generics.newCounter();
						for (String t : StrUtils.split(kwd)) {
							int w = vocab.indexOf(t.replace("/", " / "));
							if (w != -1) {
								k.add(w);
								c.incrementCount(w, 1);
							}
						}
						k.trimToSize();
						qv1 = new SparseVector(c);
						qv2 = TermWeighting.tfidf(qv1, vocab);
					}

					DenseVector ek = new DenseVector(E.colSize());

					{
						Set<Integer> toks = Generics.newHashSet();

						for (int t : qv1.indexes()) {
							toks.add(t);
						}

						Map<Integer, DenseVector> m = getEmbeddings(toks, E);

						double cnt_sum = 0;
						for (Entry<Integer, DenseVector> e : m.entrySet()) {
							int t = e.getKey();
							DenseVector et = e.getValue();
							double cnt = qv1.value(t);
							VectorMath.addAfterMultiply(et, cnt, ek);
							cnt_sum += cnt;
						}
						ek.multiply(1d / cnt_sum);
					}

					if (d.size() == 0 || k.size() == 0) {
						continue;
					}

					// DenseVector x = new DenseMatrix(new DenseVector[] { ek, ed }).toDenseVector();
					// VectorMath.subtract(ek, ed, ek);

					DoubleArray tmp = new DoubleArray();

					tmp.add(qv1.sum());
					tmp.add(dv1.sum());
					tmp.add(qv1.min());
					tmp.add(dv1.min());
					tmp.add(qv1.max());
					tmp.add(dv1.max());

					tmp.add(qv2.sum());
					tmp.add(dv2.sum());
					tmp.add(qv2.min());
					tmp.add(dv2.min());
					tmp.add(qv2.max());
					tmp.add(dv2.max());

					tmp.add(qv1.sum() / dv1.sum());
					tmp.add(qv2.sum() / dv2.sum());

					tmp.add(qv1.min() / dv1.min());
					tmp.add(qv1.max() / dv1.max());
					tmp.add(qv2.min() / dv2.min());
					tmp.add(qv2.max() / dv2.max());

					tmp.add(VectorMath.cosine(ek, ed));
					tmp.add(VectorMath.cosine(qv1, dv1));
					tmp.add(VectorMath.cosine(qv2, dv2));

					DenseVector tmp2 = ek.copy();

					VectorMath.addAfterMultiply(ek, 1, ed, -1, tmp2);
					tmp.addAll(tmp2.values());

					// tmp.addAll(ed.values());

					tmp.trimToSize();

					DenseVector x = new DenseVector(tmp);

					X.add(x);
					Y.add(i);
				}
			}

		}

		new DenseMatrix(X).writeObject(KPPath.KYP_DIR + "ext/X.ser.gz");
		new IntegerArray(Y).writeObject(KPPath.KYP_DIR + "ext/Y.ser.gz");
	}

}
