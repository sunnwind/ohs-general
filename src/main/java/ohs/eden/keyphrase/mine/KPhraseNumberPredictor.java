package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.SolverType;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentCollectionCreator;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.RandomAccessDenseMatrix;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.svm.wrapper.LibLinearTrainer;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class KPhraseNumberPredictor {

	public static SparseVector extractFeatures(Vocab vocab, MDocument doc) {

		Counter<Integer> c = Generics.newCounter();
		Counter<String> c2 = Generics.newCounter();

		double noun_cnt = 0;
		double verb_cnt = 0;

		for (MSentence ms : doc) {
			for (MultiToken mt : ms) {
				for (Token t : mt) {
					String word = t.get(0);
					String pos = t.get(1);
					String feat = String.format("%s_/_%s", word, pos);
					int f = vocab.indexOf(feat);
					if (f < 0) {
						continue;
					}
					c.incrementCount(f, 1);
					c2.incrementCount(pos, 1);

					if (pos.startsWith("NN")) {
						noun_cnt++;
					}

					if (pos.startsWith("V")) {
						verb_cnt++;
					}
				}
			}
		}

		double len_d = c.totalCount();
		double norm = 0;

		for (Entry<Integer, Double> e : c.entrySet()) {
			int f = e.getKey();
			double cnt = e.getValue();
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(f));
			c.setCount(f, tfidf);
			norm += tfidf * tfidf;
		}

		if (c.size() > 0) {
			double avg_tfidf = c.average();

			if (norm > 0) {
				norm = Math.sqrt(norm);
				c.scale(1f / norm);
			}

			double noun_ratio = noun_cnt / len_d;
			double verb_ratio = verb_cnt / len_d;

			c.incrementCount(vocab.indexOf("#len_d"), len_d);
			c.incrementCount(vocab.indexOf("#avg_idf"), avg_tfidf);
			c.incrementCount(vocab.indexOf("#noun_ratio"), noun_ratio);
			c.incrementCount(vocab.indexOf("#verb_ratio"), verb_ratio);
		}

		return new SparseVector(c);
	}

	public static void generateData() throws Exception {
		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		List<String> outs = Generics.newArrayList(ins.size());
		Counter<Integer> c = Generics.newCounter();

		ListMap<Integer, Integer> lm = Generics.newListMap();

		for (int i = 0; i < ins.size(); i++) {
			String line = ins.get(i);
			List<String> ps = Generics.newArrayList(line.split("\t"));
			ps = StrUtils.unwrap(ps);

			String kwdStr = ps.get(0);
			String title = ps.get(1);
			String abs = ps.get(2);

			int size = kwdStr.split(StrUtils.LINE_REP).length;
			String[] tmp = new String[] { size + "", title + StrUtils.LINE_REP + abs };
			tmp = StrUtils.wrap(tmp);

			lm.put(size, outs.size());

			outs.add(StrUtils.join("\t", tmp));

			c.incrementCount(size, 1);
		}

		FileUtils.writeStringCollectionAsText(KPPath.KP_DIR + "ext/label_data_phrs_number.txt", outs);

		System.out.println(c.toString(c.size()));
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// generateData();
		// vectorizeData();
		trainModel();

		System.out.println("process ends.");
	}

	public static void trainModel() throws Exception {
		SparseMatrix XD = new SparseMatrix(KPPath.KP_DIR + "ext/X_number.ser.gz");
		IntegerArray YD = new IntegerArray(KPPath.KP_DIR + "ext/Y_number.ser.gz");

		SparseMatrix X = new SparseMatrix();
		IntegerArray Y = new IntegerArray();

		SparseMatrix Xt = new SparseMatrix();
		IntegerArray Yt = new IntegerArray();

		int test_size = 1000;

		{
			IntegerArrayMatrix G = DataSplitter.group(YD);

			for (int i = 0; i < G.size(); i++) {
				IntegerArray locs = G.get(i);
				ArrayUtils.shuffle(locs.values());

				if (locs.size() < test_size) {
					test_size = locs.size() / 2;
				}

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

		Indexer<String> labelIdxer = Generics.newIndexer();
		Indexer<String> featIdxer = Generics.newIndexer();

		{
			Set<String> labels = Generics.newTreeSet();
			for (int y : Y) {
				labels.add(y + "");
			}
			labelIdxer = Generics.newIndexer(labels);

			Vocab vocab = DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser");
			featIdxer = Generics.newIndexer(vocab);
		}

		LibLinearTrainer trainer = new LibLinearTrainer();
		trainer.getParameter().setSolverType(SolverType.L2R_L2LOSS_SVR);
		trainer.getParameter().setEps(0.001);

		LibLinearWrapper llw = trainer.train(labelIdxer, featIdxer, X, Y);

		Model model = llw.getModel();
		Linear.saveModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt"), model);

		IntegerArray Yh = new IntegerArray(Yt.size());
		double diff = 0;

		for (int i = 0; i < Xt.size(); i++) {
			int y = Yt.get(i);
			SparseVector x = Xt.get(i);

			Feature[] fs = LibLinearWrapper.toFeatures(x, model.getNrFeature(), model.getBias());
			double yh = Linear.predict(model, fs);
			yh = Math.round(yh);
			Yh.add((int) yh);
			System.out.printf("(%d, %d, %d)\n", y, (int) yh, (int) (y - yh));
			diff += Math.abs(y - yh);
		}

		diff /= Yt.size();

		System.out.println(diff);

	}

	public static void vectorizeData() throws Exception {
		Vocab vocab = DocumentCollection.readVocab(KPPath.COL_DC_DIR + "vocab.ser");
		vocab.add("#len_d");
		vocab.add("#avg_idf");
		vocab.add("#noun_ratio");
		vocab.add("#verb_ratio");

		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data_phrs_number.txt");

		List<SparseVector> X = Generics.newArrayList(ins.size());
		List<Integer> Y = Generics.newArrayList(ins.size());

		for (String line : ins) {
			String[] ps = line.split("\t");
			ps = StrUtils.unwrap(ps);
			int label = Integer.parseInt(ps[0]);
			String content = ps[1].replace(StrUtils.LINE_REP, "\n");
			MDocument doc = MDocument.newDocument(content);
			SparseVector x = extractFeatures(vocab, doc);

			if (x.size() == 0) {
				continue;
			}

			Y.add(label);
			X.add(x);
		}

		new SparseMatrix(X).writeObject(KPPath.KP_DIR + "ext/X_number.ser.gz");
		new IntegerArray(Y).writeObject(KPPath.KP_DIR + "ext/Y_number.ser.gz");

		DocumentCollectionCreator.writeVocab(vocab, KPPath.KP_DIR + "ext/vocab_num_pred.ser");
	}

	public String delim = "  ";

	private Model model;

	private Vocab vocab;

	public KPhraseNumberPredictor(Vocab vocab, Model model) {
		this.vocab = vocab;
		this.model = model;
	}

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

	public IntegerArray predict(List<SparseVector> X) {
		IntegerArray ret = new IntegerArray(X.size());
		for (SparseVector x : X) {
			ret.add(predict(x));
		}
		return ret;
	}

	public int predict(MDocument doc) {
		SparseVector x = extractFeatures(vocab, doc);
		return predict(x);
	}

	public int predict(SparseVector x) {
		Feature[] fs = LibLinearWrapper.toFeatures(x, model.getNrFeature(), model.getBias());
		double yh = Linear.predict(model, fs);
		yh = Math.round(yh);
		return (int) yh;
	}

}
