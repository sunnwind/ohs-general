package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.search.app.RandomAccessDenseMatrix;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.Posting;
import ohs.corpus.search.index.PostingList;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.corpus.type.SimpleStringNormalizer;
import ohs.corpus.type.StringNormalizer;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayUtils;
import ohs.math.CommonMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.glove.CooccurrenceCounter;
import ohs.ml.glove.GloveModel;
import ohs.ml.glove.GloveParam;
import ohs.ml.glove.GloveTrainer;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.NeuralNetParams;
import ohs.ml.neuralnet.com.NeuralNetTrainer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.nonlinearity.Tanh;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListList;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.ByteSize;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class DataHandler {

	public static List<String> getKeywords(String keywordStr) {
		List<String> ret = Generics.newArrayList();
		for (String kw : keywordStr.split(";")) {
			kw = kw.trim();
			if (kw.length() > 0) {
				ret.add(kw);
			}
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DataHandler dh = new DataHandler();
		// dh.extracKeywords();
		// dh.tagPOS();
		// dh.extractKeywords();
		// dh.trainGlove();
		// dh.generateDocumentEmbedding();
		// dh.matchPhrasesToKeywords();
		// dh.getWikiPhrases();
		dh.getFrequentPhrases();
		// dh.getTrainingPhrases();

		// dh.test();
		// dh.test2();

		System.out.println("process ends.");
	}

	public void get3PKeywords() throws Exception {
		Counter<String> c = Generics.newCounter();

		for (File file : FileUtils.getFilesUnder(KPPath.COL_LINE_POS_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] parts = line.split("\t");
				parts = StrUtils.unwrap(parts);

				String kwdStr1 = parts[2];
				String kwdStr2 = parts[8];

				String[] kwds1 = kwdStr1.split(";");
				String[] kwds2 = kwdStr2.split(";");

				if (kwds1.length == 0 || kwds1.length != kwds2.length) {
					continue;
				}

				for (int i = 0; i < kwds1.length; i++) {
					String kwd1 = kwds1[i];
					String kwd2 = kwds2[i];

					if (kwd1.length() == 0 || kwd2.length() == 0) {
						continue;
					}
					kwd2 = kwd2.replace(" / ", "/").replace(" + ", " ").replace("<nl>", " ").replace(" ", "_");
					c.incrementCount(kwd2, 1);
				}
			}
		}

		double avg_len = 0;

		int max_len = 0;

		for (String kwd : c.keySet()) {
			avg_len += kwd.split(" ").length * c.getCount(kwd);
			max_len = Math.max(max_len, kwd.split(" ").length);
		}

		avg_len /= c.totalCount();

		System.out.printf("avg_len: %f\n", avg_len);
		System.out.printf("max_len: %d\n", max_len);

		FileUtils.writeStringCounterAsText(KPPath.KYP_DIR + "phrs_3p_kwds.txt.gz", c);
	}

	public void get3PKeywordsEng() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);

		Counter<String> c = Generics.newCounter();

		for (int i = 0; i < rdc.size(); i++) {
			int progress = BatchUtils.progress(i + 1, rdc.size());
			if (progress > 0) {
				System.out.printf("[%d]\n", progress);
			}
			HashMap<String, String> m = rdc.getAttrValueMap(i);
			List<String> vals = rdc.getValues(i);
			int j = 0;
			String type = vals.get(j++);
			String cn = vals.get(j++);
			String korKwsStr = vals.get(j++);
			String engKwsStr = vals.get(j++);

			if (type.equals("patent")) {
				break;
			}

			for (String kwd : engKwsStr.split(";")) {
				if (kwd.length() > 0) {
					c.incrementCount(kwd, 1);
				}
			}
		}

		FileUtils.writeStringCounterAsText(KPPath.KYP_DIR + "eng_kwds.txt", c);

		System.out.println(c);
	}

	public void getFrequentPhrases() throws Exception {
		Timer timer = Timer.newTimer();

		// String dataDir = MIRPath.TREC_CDS_2016_DIR;
		String dataDir = "../../data/naver_news/";

		PhraseCollection pc = new PhraseCollection(dataDir + "col/dc/");
		Counter<String> c = Generics.newCounter();

		int[][] ranges = BatchUtils.getBatchRanges(pc.size(), 100);
		long len = 0;

		for (int i = 0, j = 0; i < ranges.length; i++) {
			int[] range = ranges[i];

			List<SPostingList> pls = pc.getPostingLists(range[0], range[1]);
			for (SPostingList pl : pls) {
				String phrs = pl.getPhrase();
				c.incrementCount(phrs, pl.getPosData().sizeOfEntries());

				len += phrs.length() + Double.BYTES;

				j++;
				int prog = BatchUtils.progress(j, pc.size());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d, %s, %s]\n", prog, j, pc.size(), new ByteSize(len).toString(), timer.stop());
				}
			}
		}

		FileUtils.writeStringCounterAsText(dataDir + "phrs/phrs_freq.txt", c);
	}

	private String getText(List<List<List<Pair<String, String>>>> result) {
		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < result.size(); i++) {
			List<List<Pair<String, String>>> ll = result.get(i);
			for (int j = 0; j < ll.size(); j++) {
				List<Pair<String, String>> l = ll.get(j);

				for (int k = 0; k < l.size(); k++) {
					Pair<String, String> pair = l.get(k);
					String f = pair.getFirst().replace(" ", "_");
					String s = pair.getSecond();

					if (s.length() == 0) {
						continue;
					}

					sb.append(String.format("%s%s%s", f, Token.DELIM_TOKEN, s));
					// sb.append(String.format("%s%s%s", f, "/", s));

					if (k != l.size() - 1) {
						// sb.append("+");
						sb.append(MultiToken.DELIM_MULTI_TOKEN);
					}
				}

				if (j != ll.size() - 1) {
					sb.append("\n");
				}
			}
			if (i != ll.size() - 1) {
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	public void getTrainingPhrases() throws Exception {
		Counter<String> c1 = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(MIRPath.WIKI_DIR + "phrs_title.txt")) {
			String[] parts = line.split("\t");
			c1.incrementCount(parts[0].toLowerCase(), Double.parseDouble(parts[1]));
		}

		Counter<String> c2 = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(MIRPath.WIKI_DIR + "phrs_link.txt")) {
			String[] parts = line.split("\t");
			String phrs = parts[0].toLowerCase();
			double cnt = Double.parseDouble(parts[1]);
			c2.incrementCount(phrs, cnt);
		}

		String dir = MIRPath.TREC_CDS_2016_DIR;

		Counter<String> c3 = Generics.newCounter();

		for (String line : FileUtils.readLinesFromText(dir + "phrs/freq_phrss.txt")) {
			String[] parts = line.split("\t");
			c3.incrementCount(parts[0].toLowerCase(), Double.parseDouble(parts[1]));
		}

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (String phrs : c3.keySet()) {
			double cnt = c3.getCount(phrs);
			phrs = phrs.replace("_", " ");
			int len_phrs = phrs.split(" ").length;
			int type = 0;

			if (c1.containsKey(phrs)) {

			} else {
				if (c2.containsKey(phrs)) {
					type = 1;
				} else {
					type = 2;
				}
			}

			if (type == 0) {
				cm.incrementCount("GOOD", phrs, cnt);
			} else if (type == 1) {
				cm.incrementCount("NotBad", phrs, cnt);
			} else if (type == 2) {
				cm.incrementCount("BAD", phrs, cnt);
			}
		}

		FileUtils.writeStringCounterAsText(dir + "phrs/phrs_good.txt", cm.getCounter("GOOD"));
		FileUtils.writeStringCounterAsText(dir + "phrs/phrs_not_bad.txt", cm.getCounter("NotBad"));
		FileUtils.writeStringCounterAsText(dir + "phrs/phrs_bad.txt", cm.getCounter("BAD"));
	}

	public void getWikiPhrases() throws Exception {
		Timer timer = Timer.newTimer();
		RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.WIKI_COL_DC_DIR);

		int[][] ranges = BatchUtils.getBatchRanges(rdc.size(), 1000);

		Counter<String> c1 = Generics.newCounter();
		Counter<String> c2 = Generics.newCounter();

		String reg = "\\([^\\(\\)]+\\)";

		Pattern p = Pattern.compile(reg);

		StringNormalizer sn = new SimpleStringNormalizer(true);

		for (int i = 0; i < ranges.length; i++) {
			int[] range = ranges[i];

			ListList<String> ps = rdc.getValues(range);

			for (int j = 0; j < ps.size(); j++) {
				List<String> vals = ps.get(j);

				String title = vals.get(2);
				String phrss = vals.get(4);

				Matcher m = p.matcher(title);
				StringBuffer sb = new StringBuffer();

				if (m.find()) {
					String g = m.group();
					// System.out.println(g);
					// int s = m.start();
					m.appendReplacement(sb, "");
				}
				m.appendTail(sb);

				title = sb.toString().trim();

				if (title.startsWith("List of")) {
					title = title.substring("List of".length() + 1);
				}

				title = sn.normalize(title);

				if (title.split(" ").length > 1 && !title.contains("?")) {
					c1.incrementCount(title, 1);
				}

				for (String phrs : phrss.split("\\|")) {
					// phrs = StrUtils.join(" ", NLPUtils.tokenize(title));
					phrs = sn.normalize(phrs);
					if (phrs.split(" ").length > 1 && !phrs.contains("?")) {
						c2.incrementCount(phrs, 1);
					}
				}
			}

			int prog = BatchUtils.progress(i + 1, ranges.length);

			if (prog > 0) {
				System.out.printf("[%d percent, %d/%d, %s]\n", prog, i + 1, ranges.length, timer.stop());
			}
		}

		FileUtils.writeStringCounterAsText(MIRPath.WIKI_DIR + "phrs_title.txt", c1);
		FileUtils.writeStringCounterAsText(MIRPath.WIKI_DIR + "phrs_link.txt", c2);
	}

	public void matchPhrasesToKeywords() throws Exception {
		Counter<String> c = FileUtils.readStringCounterFromText(KPPath.KYP_DIR + "phrs_3p_kwds.txt.gz");

		List<String> lines = FileUtils.readLinesFromText(KPPath.KYP_DIR + "phrs_3p.txt.gz");

		Counter<String> cc = Generics.newCounter();

		for (int i = 0; i < lines.size(); i++) {
			List<String> parts = Generics.newArrayList();

			for (String s : lines.get(i).split("\t")) {
				parts.add(s);
			}

			int cnt = (int) c.getCount(parts.get(0));

			if (cnt > 0) {
				cc.incrementCount("Phrs+Kwd", 1);
			}

			cc.incrementCount("Phrs", 1);

			parts.add(cnt + "");

			lines.set(i, StrUtils.join("\t", parts));
		}

		System.out.println(cc);

		FileUtils.writeStringCollection(KPPath.KYP_DIR + "phrs_3p_label.txt.gz", lines);
	}

	public void queryGloveModel() throws Exception {
		String dir = KPPath.DATA_DIR;
		//
		// GloveModel M = new GloveModel();
		// M.readObject(dir + "glove_model_noun.ser.gz");
		//
		// DocumentCollection lsc = new DocumentCollection(dir +
		// "col/noun_dc/");
		//
		// WordVectorModel vecs = new WordVectorModel(lsc.getVocab(),
		// M.getAveragedModel());
		//
		// Set<String> stopwords =
		// FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		//
		// WordSearcher.interact(new WordSearcher(vecs, stopwords));
		System.out.println("process ends.");
	}

	public void tagPOS() throws Exception {
		Komoran komoran = new Komoran("lib/models-full/");

		FileUtils.deleteFilesUnder(KPPath.COL_LINE_POS_DIR);

		for (File file : FileUtils.getFilesUnder(KPPath.COL_LINE_DIR)) {
			List<String> lines = FileUtils.readLinesFromText(file);

			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);

				List<String> parts = Generics.newArrayList(line.split("\t"));

				parts = StrUtils.unwrap(parts);

				String type = parts.get(0);
				String cn = parts.get(1);
				String korKwdStr = parts.get(2);
				String engKwdStr = parts.get(3);
				String korTitle = parts.get(4);
				String engTitle = parts.get(5);
				String korAbs = parts.get(6);
				String engAbs = parts.get(7);

				// if (!cn.equals("JAKO199910102414471")) {
				// continue;
				// }

				List<String> korKwds = getKeywords(korKwdStr);
				List<String> engKwds = getKeywords(engKwdStr);

				if (korKwds.size() > 0) {
					List<String> korKwds2 = Generics.newArrayList(korKwds.size());
					for (int j = 0; j < korKwds.size(); j++) {
						String kwd = korKwds.get(j);
						kwd = getText(komoran.analyze(kwd, 1));
						korKwds2.add(kwd);
					}

					parts.add(StrUtils.join(";", korKwds2).replace("\n", StrUtils.LINE_REP));
				} else {
					parts.add("");
				}

				if (korTitle.length() > 0) {
					String t = getText(komoran.analyze(korTitle, 1));
					parts.add(t.replace("\n", StrUtils.LINE_REP));
				} else {
					parts.add("");
				}

				if (korAbs.length() > 0) {
					StringBuffer sb = new StringBuffer();
					for (String sent : korAbs.replace(". ", ".\n").split("\n")) {
						sb.append(getText(komoran.analyze(sent, 1)));
						sb.append("\n\n");
					}

					String t = sb.toString().trim();
					parts.add(t.replace("\n", StrUtils.LINE_REP));
				} else {
					parts.add("");
				}

				parts = StrUtils.wrap(parts);

				lines.set(i, StrUtils.join("\t", parts));
			}

			FileUtils.writeStringCollection(file.getPath().replace("line", "line_pos"), lines);
		}
	}

	public void test() throws Exception {
		System.out.println("process begins.");

		PhraseCollection pc = new PhraseCollection(MIRPath.OHSUMED_COL_DC_DIR);
		DocumentCollection dc = new DocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);
		InvertedIndex ii = new InvertedIndex(MIRPath.OHSUMED_COL_DC_DIR);
		Vocab vocab = dc.getVocab();

		System.out.println(pc.size());

		Counter<Integer> c = Generics.newCounter();

		for (int len : pc.getPhraseLengths()) {
			c.incrementCount(len, 1);
		}

		System.out.println(c.toString());

		for (int i = 0; i < pc.size(); i++) {
			int phrs_len = pc.getPhraseLengths().get(i);

			if (phrs_len != 2) {
				continue;
			}

			SPostingList pl = pc.getPostingList(i);
			System.out.println(pl.toString());

			String phrs = pl.getPhrase();

			int len_phrs = phrs.split("_").length;

			String[] words = phrs.split("_");
			IntegerArray Q = dc.getVocab().indexesOf(words);

			// if (Q.size() == 2) {
			// continue;
			// }

			for (int k = 1; k < Q.size(); k++) {
				String prefix = StrUtils.join("_", words, 0, k);
				String suffix = StrUtils.join("_", words, k, Q.size());

				IntegerArray Q2 = vocab.indexesOf(prefix.split("_"));
				IntegerArray Q3 = vocab.indexesOf(suffix.split("_"));

				PostingList pl_x = ii.getPostingList(Q2, true, 1);
				PostingList pl_y = ii.getPostingList(Q3, true, 1);
				PostingList pl_xy = ii.getPostingList(Q, true, 1);

				double pmi = CommonMath.pmi(pl_x.size(), pl_y.size(), pl_xy.size(), dc.size(), true);

				System.out.printf("[%s, %s, %f]\n", prefix, suffix, pmi);
			}
			System.out.println();

		}
	}

	public void test2() throws Exception {
		System.out.println("process begins.");

		CounterMap<String, String> cm = Generics.newCounterMap();

		cm.setCounter("POS", FileUtils.readStringCounterFromText(MIRPath.OHSUMED_DIR + "phrs_train/pos.txt"));
		cm.setCounter("NEG", FileUtils.readStringCounterFromText(MIRPath.OHSUMED_DIR + "phrs_train/neg.txt"));

		PhraseCollection pc = new PhraseCollection(MIRPath.OHSUMED_COL_DC_DIR);
		DocumentCollection dc = new DocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);

		System.out.println(pc.size());

		Counter<String> posPhrss = cm.getCounter("POS");

		for (int i = 0; i < pc.size(); i++) {
			SPostingList pl = pc.getPostingList(i);
			System.out.println(pl.toString());

			String phrs = pl.getPhrase();

			if (!posPhrss.containsKey(phrs)) {
				continue;
			}

			int len_phrs = phrs.split("_").length;

			IntegerArray dseqs = pl.getDocseqs();
			IntegerArrayMatrix posData = pl.getPosData();

			for (int j = 0; j < pl.size(); j++) {
				Posting p = pl.getPosting(j);

				int dseq = p.getDocseq();
				IntegerArray poss = p.getPoss();

				List<String> words = dc.getWords(dseq);
				int window_size = 4;

				for (int start : poss) {
					int end = start + len_phrs;
					int left = Math.max(0, start - window_size);
					int right = Math.min(end + window_size, words.size());

					List<String> context = Generics.newArrayList();

					for (int k = left; k < right; k++) {
						String word = words.get(k);
						if (k == start) {
							context.add("[");
						} else if (k == end) {
							context.add("]");
						}
						context.add(word);
					}

					System.out.println(StrUtils.join(" ", context));
				}
			}

			System.out.println();

		}
	}

	public void trainGlove() throws Exception {
		String dir = KPPath.COL_DIR;
		String scDir = dir + "dc/";
		String ccDir = dir + "cocnt/";
		String outFileName1 = KPPath.KYP_DIR + "glove_model.ser.gz";
		String outFileName2 = KPPath.KYP_DIR + "glove_embedding.ser.gz";

		int thread_size = 50;
		int hidden_size = 200;
		int max_iters = 30;
		int window_size = 10;
		double learn_rate = 0.001;
		boolean use_adam = true;
		boolean read_all_files = true;

		int batch_size = 100;

		DocumentCollection dc = new DocumentCollection(scDir);

		Vocab vocab = dc.getVocab();

		if (!FileUtils.exists(ccDir)) {
			CooccurrenceCounter cc = new CooccurrenceCounter(scDir, null);
			cc.setWindowSize(window_size);
			cc.setCountThreadSize(thread_size);
			cc.setSymmetric(true);
			cc.setOutputFileSize(batch_size);
			cc.count();
		}

		GloveParam param = new GloveParam(vocab.size(), hidden_size);
		param.setThreadSize(thread_size);
		param.setLearnRate(learn_rate);

		GloveTrainer trainer = new GloveTrainer();
		GloveModel M = trainer.train(param, vocab, ccDir, max_iters, read_all_files, use_adam);
		M.writeObject(outFileName1);

		DenseMatrix E = M.getAveragedModel();
		E.writeObject(outFileName2);
	}

}
